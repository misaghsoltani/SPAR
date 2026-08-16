"""Load model bundles and dispatch discrete, continuous, or combined evaluation."""

from __future__ import annotations

from logging import getLogger
import pathlib
from typing import TYPE_CHECKING

import torch

from spar.data.testing_dataset import TestDataLoader, create_dataloader
from spar.utils.log_utils.console_logger import terminal_console as console
from spar.utils.log_utils.wandb_logger import reset_active_tracking_session, set_active_tracking_session
from spar.utils.pytorch_utils.nnet_utils import load_model

from .base_tester_new import CombinedModelTester, ModelBundle
from .continuous_model_tester import ContinuousModelTester
from .discrete_model_tester import DiscreteModelTester

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextvars import Token
    from logging import Logger
    from typing import TypeAlias

    from torch import nn

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import (
        CompileConfig,
        ModelConfig,
        PretrainedModelTestPathConfig,
        TestConfig,
        TestDataLoaderConfig,
        TestDataPathConfig,
        TestModelSPARConfig,
        TestSavePathConfig,
    )

    from .base_tester import EvaluateResults as SeparateEvaluateResults
    from .base_tester_new import EvaluateResults as CombinedEvaluateResults

    EvaluateResults: TypeAlias = SeparateEvaluateResults | CombinedEvaluateResults
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession


logger: Logger = getLogger(__name__)


def load_bundle(
    env: ABCEnvironment[ABCState],
    model_type: str,
    model_cfg: ModelConfig,
    compile_cfg: CompileConfig,
    paths: PretrainedModelTestPathConfig,
    device: str,
    use_alignment_model: bool,
    end_to_end: bool,
) -> ModelBundle:
    """Load a model bundle (encoder, transition, decoder, alignment) from pretrained paths."""
    get_encoder: Callable[[ModelConfig], nn.Module] = env.get_encoder_disc
    get_decoder: Callable[[ModelConfig], nn.Module] = env.get_decoder_disc
    get_env_model: Callable[[ModelConfig], nn.Module] = env.get_env_model_disc

    if model_type == "discrete":
        get_encoder = env.get_encoder_disc
        get_decoder = env.get_decoder_disc
        get_env_model = env.get_env_model_disc
    else:
        get_encoder = env.get_encoder_cont
        get_decoder = env.get_decoder_cont
        get_env_model = env.get_env_model_cont

    # Transition
    assert paths.transition_model_path is not None, "Pretrained transition model path must be provided."
    transition_model: nn.Module = load_model(
        model=get_env_model(model_cfg),
        device=device,
        pretrained_path=paths.transition_model_path,
        freeze=True,
        compile_cfg=compile_cfg,
    )
    # Decoder
    assert paths.decoder_path is not None, "Pretrained decoder path must be provided."
    decoder: nn.Module = load_model(
        model=get_decoder(model_cfg),
        device=device,
        pretrained_path=paths.decoder_path,
        freeze=True,
        compile_cfg=compile_cfg,
    )
    # Alignment (optional)
    alignment_model: nn.Module | None = None
    if use_alignment_model and paths.alignment_model_path:
        alignment_model = load_model(
            model=env.get_alignment_model(model_cfg),
            device=device,
            pretrained_path=paths.alignment_model_path,
            freeze=True,
            compile_cfg=compile_cfg,
        )
    # Encoder (or alignment in end-to-end continuous mode fallback)
    enc_path: str | None = paths.encoder_path
    if end_to_end and model_type == "continuous" and enc_path is None:
        enc_path = paths.alignment_model_path
        get_encoder = env.get_alignment_model
    encoder: nn.Module = load_model(
        model=get_encoder(model_cfg), device=device, pretrained_path=enc_path, freeze=True, compile_cfg=compile_cfg
    )
    return ModelBundle(encoder=encoder, transition=transition_model, decoder=decoder, alignment=alignment_model)


def run_test(
    env: ABCEnvironment[ABCState], cfg: TestModelSPARConfig, tracking: WandbTrackingSession | None = None
) -> EvaluateResults:
    """Test a trained model using the specified configuration.

    Args:
        env: Environment instance.
        cfg: Evaluation configuration.
        tracking: Optional W&B tracking session. When provided, it becomes the
            active session for the duration of the evaluation so that metric
            logging inside the testers reaches the caller's run.

    Returns:
        Dictionary containing evaluation results.
    """
    if tracking is None:
        return _run_test_impl(env, cfg)

    token: Token[WandbTrackingSession | None] = set_active_tracking_session(tracking)
    try:
        return _run_test_impl(env, cfg)
    finally:
        reset_active_tracking_session(token)


def _run_test_impl(env: ABCEnvironment[ABCState], cfg: TestModelSPARConfig) -> EvaluateResults:
    """Run the model evaluation described by the configuration.

    Args:
        env: Environment instance.
        cfg: Evaluation configuration.

    Returns:
        Dictionary containing evaluation results.
    """
    # Set device
    test_cfg: TestConfig = cfg.test
    compile_cfg: CompileConfig = test_cfg.compile
    end_to_end: bool = test_cfg.end_to_end
    model_cfg: ModelConfig = cfg.model
    pretrained_paths_cfg: PretrainedModelTestPathConfig = cfg.pretrained_model_paths
    data_paths_cfg: TestDataPathConfig = cfg.data_paths
    save_paths_cfg: TestSavePathConfig = cfg.save_paths

    device: str = test_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("[bold orange]WARNING:[/ bold orange] CUDA is not available, switching to CPU.")
        device = "cpu"

    is_continuous: bool = cfg.test_model_type == "continuous"
    use_combined: bool = test_cfg.tester_mode.lower() == "combined"

    # For mode, pick single set, for combined, load both
    get_encoder: Callable[[ModelConfig], nn.Module] = env.get_encoder_cont if is_continuous else env.get_encoder_disc
    get_decoder: Callable[[ModelConfig], nn.Module] = env.get_decoder_cont if is_continuous else env.get_decoder_disc
    get_env_model: Callable[[ModelConfig], nn.Module] = (
        env.get_env_model_cont if is_continuous else env.get_env_model_disc
    )

    use_wandb: bool = cfg.wandb.mode in {"online", "offline"}
    # Create dataloader (encoder only needed if encoded targets requested, avoid mismatch in combined mode)
    dataloader_cfg: TestDataLoaderConfig = test_cfg.dataloader
    with console.status("Loading testing data...\n", spinner="dots"):
        dataloader: TestDataLoader | None = None
        if use_combined:
            dataloader = create_dataloader(
                file_path=data_paths_cfg.test_data,
                batch_size=dataloader_cfg.batch_size,
                encoder=None,
                use_encoded_targets=False,
                precompute_targets=False,
                device=device,
                use_variation_for_all_states=dataloader_cfg.use_variation_for_all_states,
                enable_memory_optimization=dataloader_cfg.enable_memory_optimization,
                variations_to_use=dataloader_cfg.variations_to_use,
                variations_to_ignore=dataloader_cfg.variations_to_ignore,
            )

    # Create save directory
    pathlib.Path(save_paths_cfg.images_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(save_paths_cfg.plots_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(save_paths_cfg.metrics_dir).mkdir(exist_ok=True, parents=True)

    evaluator: ContinuousModelTester | DiscreteModelTester | CombinedModelTester
    results: EvaluateResults

    if use_combined:
        # Load both discrete and continuous bundles
        disc_paths: PretrainedModelTestPathConfig | None = cfg.pretrained_model_paths_discrete
        cont_paths: PretrainedModelTestPathConfig | None = cfg.pretrained_model_paths_continuous
        if disc_paths is None or cont_paths is None:
            raise ValueError(
                "tester_mode='combined' requires both 'pretrained_model_paths_discrete' and "
                "'pretrained_model_paths_continuous' in the config."
            )
        disc_bundle: ModelBundle = load_bundle(
            env=env,
            model_type="discrete",
            model_cfg=model_cfg,
            compile_cfg=compile_cfg,
            paths=disc_paths,
            device=device,
            use_alignment_model=test_cfg.use_alignment_model,
            end_to_end=end_to_end,
        )
        cont_bundle: ModelBundle = load_bundle(
            env=env,
            model_type="continuous",
            model_cfg=model_cfg,
            compile_cfg=compile_cfg,
            paths=cont_paths,
            device=device,
            use_alignment_model=test_cfg.use_alignment_model,
            end_to_end=end_to_end,
        )

        # Prefer per-model metric lists when present, otherwise fall back to the unified list
        metrics_disc_cfg: list[str] | None = test_cfg.metrics_to_save_discrete or test_cfg.metrics_to_save
        metrics_cont_cfg: list[str] | None = test_cfg.metrics_to_save_continuous or test_cfg.metrics_to_save

        evaluator = CombinedModelTester(
            discrete=disc_bundle,
            continuous=cont_bundle,
            device=device,
            output_dir=save_paths_cfg.images_dir,
            variations_to_use=dataloader_cfg.variations_to_use,
            variations_to_ignore=dataloader_cfg.variations_to_ignore,
            use_variation_for_all_states=dataloader_cfg.use_variation_for_all_states,
            save_interval=test_cfg.save_interval,
            visualization_format=test_cfg.visualization_format,
            visualization_episode_index=test_cfg.visualization_episode_index,
            visualization_steps=test_cfg.visualization_steps,
            log_interval=test_cfg.log_interval,
            apply_diff_highlighting=test_cfg.apply_diff_highlighting,
            row_labels=test_cfg.row_labels or ["Discrete Recon", "Continuous Recon", "Ground Truth"],
            variant_panel_title=test_cfg.variant_panel_title,
            suptitle_cfg=test_cfg.suptitle,
            metrics_to_save_discrete=metrics_disc_cfg,
            metrics_to_save_continuous=metrics_cont_cfg,
            column_metric_priority=test_cfg.column_metric_priority,
        )
        if dataloader is None:
            raise RuntimeError("Combined evaluator dataloader was not initialized")
        results = evaluator.evaluate_dataloader(dataloader)
        logger.info("Combined evaluation completed.")
        return results

    # Load single bundle according to test_model_type
    single_paths: PretrainedModelTestPathConfig = pretrained_paths_cfg
    # Load transition/decoder/alignment/encoder
    # Transition
    assert single_paths.transition_model_path is not None, "Pretrained transition model path must be provided."
    transition_model: nn.Module = load_model(
        model=get_env_model(model_cfg),
        device=device,
        pretrained_path=single_paths.transition_model_path,
        freeze=True,
        compile_cfg=compile_cfg,
    )
    # Decoder
    assert single_paths.decoder_path is not None, "Pretrained decoder path must be provided."
    decoder: nn.Module = load_model(
        model=get_decoder(model_cfg),
        device=device,
        pretrained_path=single_paths.decoder_path,
        freeze=True,
        compile_cfg=compile_cfg,
    )
    # Alignment
    alignment_model: nn.Module | None = None
    pretrained_align_model_path: str | None = single_paths.alignment_model_path
    if test_cfg.use_alignment_model and pretrained_align_model_path:
        alignment_model = load_model(
            model=env.get_alignment_model(model_cfg),
            device=device,
            pretrained_path=pretrained_align_model_path,
            freeze=True,
            compile_cfg=compile_cfg,
        )
    # Encoder (or alignment for end-to-end)
    pretrained_encoder_path: str | None = single_paths.encoder_path
    if end_to_end and cfg.test_model_type == "continuous" and single_paths.encoder_path is None:
        pretrained_encoder_path = pretrained_align_model_path
        get_encoder = env.get_alignment_model
    encoder: nn.Module = load_model(
        model=get_encoder(model_cfg),
        device=device,
        pretrained_path=pretrained_encoder_path,
        freeze=True,
        compile_cfg=compile_cfg,
    )

    # Build dataloader (uses encoder if encoded targets are requested)
    with console.status("Loading testing data...\n", spinner="dots"):
        dataloader = create_dataloader(
            file_path=data_paths_cfg.test_data,
            batch_size=dataloader_cfg.batch_size,
            encoder=encoder,
            use_encoded_targets=dataloader_cfg.use_encoded_targets,
            precompute_targets=dataloader_cfg.precompute_targets,
            device=device,
            use_variation_for_all_states=dataloader_cfg.use_variation_for_all_states,
            enable_memory_optimization=dataloader_cfg.enable_memory_optimization,
            variations_to_use=dataloader_cfg.variations_to_use,
            variations_to_ignore=dataloader_cfg.variations_to_ignore,
        )

    encoder_for_eval: nn.Module = (
        alignment_model if test_cfg.use_alignment_model and alignment_model is not None else encoder
    )

    # Create evaluator (two-row tester) by selecting class dynamically
    tester_cls: type[ContinuousModelTester | DiscreteModelTester] = (
        ContinuousModelTester if is_continuous else DiscreteModelTester
    )
    evaluator = tester_cls(
        encoder=encoder,
        transition_model=transition_model,
        decoder=decoder,
        alignment_model=encoder_for_eval,
        device=device,
        test_model_type=cfg.test_model_type,
        use_alignment_model=test_cfg.use_alignment_model,
        output_dir=save_paths_cfg.images_dir,
        variations_to_use=dataloader_cfg.variations_to_use,
        variations_to_ignore=dataloader_cfg.variations_to_ignore,
        use_variation_for_all_states=dataloader_cfg.use_variation_for_all_states,
        save_interval=test_cfg.save_interval,
        visualization_format=test_cfg.visualization_format,
        visualization_episode_index=test_cfg.visualization_episode_index,
        visualization_steps=test_cfg.visualization_steps,
        log_interval=test_cfg.log_interval,
        use_wandb=use_wandb,
        end_to_end=end_to_end,
        top_k=test_cfg.top_k,
        apply_diff_highlighting=test_cfg.apply_diff_highlighting,
        metrics_to_save=test_cfg.metrics_to_save,
        row_labels=test_cfg.row_labels,
        rightmost_col_row_labels=test_cfg.rightmost_col_row_labels,
        rightmost_col_row_labels_side=test_cfg.rightmost_col_row_labels_side or "right",
        variant_panel_title=test_cfg.variant_panel_title,
        suptitle_cfg=test_cfg.suptitle,
        column_metric_priority=test_cfg.column_metric_priority,
    )

    # Run evaluation
    results = evaluator.evaluate_dataloader(dataloader)
    logger.info("Evaluation completed.")

    return results
