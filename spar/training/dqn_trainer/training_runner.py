"""DQN training runner with config-based workflow."""

from __future__ import annotations

from logging import getLogger
import pathlib
import pickle
import time
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
import torch.distributed.distributed_c10d as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from spar.models.factory import ModelFactory
from spar.training.dqn_trainer.base_trainer import (
    _compile_module_if_enabled,
    copy_files,
    distributed_barrier_sync,
    evaluate_greedy_policy,
    get_device,
    get_time_str,
    greedy_policy_rollout,
    load_data,
    run_data_gen,
    run_train,
    save_distributed_checkpoint,
    setup_distributed_training,
    setup_nccl_environment,
)
from spar.utils.log_utils.wandb_logger import get_active_tracking_session, log_metrics
from spar.utils.pytorch_utils.nnet_utils import load_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger
    from typing import TypeAlias

    from numpy.typing import NDArray
    from torch import nn
    from torch.optim.optimizer import Optimizer

    from spar.environments.abstracts.environment import ABCEnvironment
    from spar.environments.abstracts.state import ABCState
    from spar.training.dqn_trainer.base_trainer import DataQueueType, TimeQueueType
    from spar.utils.config_utils.config_schema import (
        CompileConfig,
        DQNTrainConfig,
        ModelConfig,
        PretrainedModelPathConfig,
        TrainConfig,
        TrainHeuristicSPARConfig,
    )
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession


logger: Logger = getLogger(__name__)
ReporterScalar: TypeAlias = float | int | bool | str | None
ReporterPayload: TypeAlias = dict[str, ReporterScalar | dict[str, ReporterScalar]]


class HeuristicTrainResult(TypedDict):
    """Structured return value for heuristic training."""

    metrics: dict[str, float | int]
    artifacts: dict[str, str]


# from spar.training.dqn_trainer.base_trainer import create_fsdp_model


def train_heuristic(
    env: ABCEnvironment[ABCState],
    cfg: TrainHeuristicSPARConfig,
    tracking: WandbTrackingSession | None = None,
    reporter: Callable[[ReporterPayload], None] | None = None,
) -> HeuristicTrainResult:
    """Main function to run DQN heuristic training using config-based workflow.

    Args:
        env: Environment instance.
        cfg: Validated configuration for heuristic training.
        tracking: Optional W&B tracking session used for metric logging. Falls back
            to the active managed session when not provided.
        reporter: Optional sparse progress callback invoked at evaluation checkpoints.

    Returns:
        HeuristicTrainResult: Final metrics and artifact directories for the trained DQN.
    """
    train_cfg: TrainConfig = cfg.train
    if train_cfg.dqn is None:
        raise ValueError("DQN training config is required at train.dqn.")
    dqn_cfg: DQNTrainConfig = train_cfg.dqn
    compile_cfg: CompileConfig = train_cfg.compile
    model_cfg: ModelConfig = cfg.model
    pretrained_paths_cfg: PretrainedModelPathConfig = cfg.pretrained_model_paths

    configured_device: str = train_cfg.device
    if configured_device == "cuda" and not torch.cuda.is_available():
        logger.warning("[bold orange]WARNING:[/ bold orange] CUDA is not available, switching to CPU.")
        configured_device = "cpu"

    # Get model factory functions
    get_dqn_model = env.get_dqn
    get_env_model = env.get_env_model_disc
    get_encoder = env.get_encoder_disc

    dqn: nn.Module = load_model(
        model=get_dqn_model(model_cfg),
        device=configured_device,
        pretrained_path=None,
        freeze=False,
        strip_compiled_prefixes=False,
        compile_cfg=compile_cfg,
    )

    # Load pre-trained models
    pretrained_encoder_path: str | None = pretrained_paths_cfg.encoder_path
    assert pretrained_encoder_path is not None, "Pretrained encoder path must be provided."
    load_model(
        model=get_encoder(model_cfg),
        device=configured_device,
        pretrained_path=pretrained_encoder_path,
        freeze=True,
        strip_compiled_prefixes=False,
        compile_cfg=compile_cfg,
    )

    pretrained_trans_model_path: str | None = pretrained_paths_cfg.transition_model_path
    assert pretrained_trans_model_path is not None, "Pretrained transition model path must be provided."
    transition_model: nn.Module = load_model(
        model=get_env_model(model_cfg),
        device=configured_device,
        pretrained_path=pretrained_trans_model_path,
        freeze=True,
        strip_compiled_prefixes=False,
        compile_cfg=compile_cfg,
    )

    # Set NCCL variables before creating the process group.
    setup_nccl_environment(dqn_cfg.nccl)

    # Initialize distributed training if available
    rank, world_size, local_rank = setup_distributed_training()
    is_distributed = world_size > 1

    # Get device
    runtime_device, devices, on_gpu = get_device()

    # Set device for distributed training
    if is_distributed and torch.cuda.is_available():
        runtime_device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(runtime_device)

    if rank == 0:
        logger.info(f"device: {runtime_device}, devices: {devices}, on_gpu: {on_gpu}")

    # Setup directories
    model_dir = f"{dqn_cfg.save_dir}/{dqn_cfg.nnet_name}/"
    targ_dir = f"{model_dir}/target/"
    curr_dir = f"{model_dir}/current/"

    # Create directories
    for directory in [targ_dir, curr_dir]:
        if not pathlib.Path(directory).exists():
            pathlib.Path(directory).mkdir(parents=True)

    if rank == 0:
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Batch size: {cfg.train.batch_size}")

    # Resolve the W&B tracking session (rank gating is handled by the session itself)
    tracking_session: WandbTrackingSession | None = tracking if tracking is not None else get_active_tracking_session()

    # # Wrap environment model with FSDP if distributed and enabled
    # if is_distributed and dqn_cfg.fsdp.enabled:
    #     transition_model = create_fsdp_model(transition_model, fsdp_cfg=dqn_cfg.fsdp)

    # Load offline data
    if rank == 0:
        logger.info("Loading offline data")

    with pathlib.Path(dqn_cfg.val_data_path).open("rb") as f:
        episodes = pickle.load(f)
    states_offline_np: NDArray[np.float32] = np.concatenate(episodes[0], axis=0)

    # Load DQN
    if rank == 0:
        logger.info("Getting DQN")

    dqn, itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best = load_data(
        env, transition_model, states_offline_np, runtime_device, dqn_cfg, curr_dir, model_cfg
    )

    dqn.to(runtime_device)
    dqn = _compile_module_if_enabled(dqn, compile_cfg)

    use_fused_adam = (
        train_cfg.optimizer == "adam"
        and runtime_device.type == "cuda"
        and hasattr(torch.optim.Adam, "_fused_implementation")
        and dqn_cfg.optimization.fused_optimizer
    )
    if use_fused_adam:
        optimizer: Optimizer = torch.optim.Adam(dqn.parameters(), lr=dqn_cfg.lr, fused=True)
    else:
        optimizer = ModelFactory.build_optimizer(
            optimizer_name=train_cfg.optimizer, params=list(dqn.parameters()), lr=dqn_cfg.lr
        )
    ModelFactory.build_scheduler(optimizer=optimizer, cfg=train_cfg.scheduler)

    # # Wrap DQN with FSDP for distributed training if enabled
    # if is_distributed and dqn_cfg.fsdp.enabled:
    #     dqn = create_fsdp_model(dqn, fsdp_cfg=dqn_cfg.fsdp)

    last_loss: float = float("nan")
    per_solved_fixed: float = 0.0
    per_solved_test: float = 0.0

    # Training loop
    while itr < dqn_cfg.max_itrs:
        max_steps = min(update_num + 1, dqn_cfg.max_solve_steps)
        assert max_steps >= 1, "max_solve_steps must be at least 1"

        # Generate data and DQN update
        if rank == 0:
            logger.info("")
        start_time: float = time.time()

        if dqn_cfg.update_itrs and len(dqn_cfg.update_itrs) > 0:
            target_train_itrs = dqn_cfg.update_itrs[update_num] - itr
        else:
            target_train_itrs = int(np.ceil(dqn_cfg.states_per_update / dqn_cfg.batch_size))

        if rank == 0:
            logger.info(f"Target train itrs: {target_train_itrs}, Max steps: {max_steps}")

        num_gen_itrs = int(np.ceil(target_train_itrs / max_steps))

        # Calculate batch size multiplier to reach update batch size
        batch_size_mult = int(np.ceil(dqn_cfg.update_nnet_batch_size / dqn_cfg.batch_size))
        batch_size_up = dqn_cfg.batch_size * batch_size_mult
        num_gen_itrs_up = int(np.ceil(num_gen_itrs / batch_size_mult))

        if rank == 0:
            logger.info(f"Generating data with batch size: {batch_size_up}, iterations: {num_gen_itrs_up}")

        res_data: tuple[DataQueueType, TimeQueueType] = run_data_gen(
            env_name=cfg.env.name,
            train_data_path=dqn_cfg.train_data_path,
            batch_size=batch_size_up,
            num_batches=num_gen_itrs_up,
            start_steps=dqn_cfg.start_steps,
            goal_steps=dqn_cfg.goal_steps,
            per_eq_tol=dqn_cfg.per_eq_tol,
            max_steps=max_steps,
            transition_model_path=pretrained_trans_model_path,
            dqn_curr_dir=curr_dir,
            dqn_targ_dir=targ_dir,
            dqn=dqn,
            comm_mode=dqn_cfg.comm_mode,
            device=runtime_device,
            model_cfg=model_cfg,
        )
        (s_start, s_goal, acts, ctgs), times = res_data

        # Train
        actual_train_itrs = int(np.ceil(s_start.shape[0] / dqn_cfg.batch_size))
        if rank == 0:
            logger.info(np.unique(ctgs.astype(int)))
            logger.info(f"Times - {get_time_str(times)}, Total: {time.time() - start_time:.2f}\n")
            logger.info(f"Training model for update number {update_num} for {actual_train_itrs} iterations")

        last_loss = run_train(
            dqn=dqn,
            optimizer=optimizer,
            states_start_np=s_start,
            states_goal_np=s_goal,
            actions_np=acts,
            ctgs=ctgs,
            batch_size=dqn_cfg.batch_size,
            device=runtime_device,
            on_gpu=on_gpu,
            num_itrs=actual_train_itrs,
            train_itr=itr,
            lr=dqn_cfg.lr,
            lr_d=dqn_cfg.lr_d,
            compile_cfg=compile_cfg,
            memory_cfg=dqn_cfg.memory,
            optimization_cfg=dqn_cfg.optimization,
            display=True,
            use_dataloader=True,
        )

        itr += actual_train_itrs

        # Save model state with distributed checkpoint support
        if isinstance(dqn, FSDP) and is_distributed:
            # Use distributed checkpointing for FSDP models
            save_distributed_checkpoint(dqn, optimizer, curr_dir, itr)
        elif rank == 0:
            if isinstance(dqn, FSDP):
                # For FSDP models, save the state dict directly with proper naming
                state_dict = {"dqn_model": dqn.state_dict()}
                torch.save(state_dict, f"{curr_dir}/model_state_dict.pt")
            else:
                # Fallback to regular save for non-FSDP models
                torch.save(dqn.state_dict(), f"{curr_dir}/model_state_dict.pt")

        # Evaluation phase with inference_mode
        start_time = time.time()
        dqn.eval()
        transition_model.eval()
        max_gbfs_steps = min(update_num + 1, dqn_cfg.goal_steps)

        if rank == 0:
            logger.info(f"\nTesting with {max_gbfs_steps} GBFS steps\nFixed test states ({states_start_t_np.shape[0]})")

        # Evaluation does not require autograd state.
        with torch.inference_mode():
            is_solved_fixed, _ = greedy_policy_rollout(
                dqn,
                transition_model,
                states_start_t_np,
                states_goal_t_np,
                dqn_cfg.per_eq_tol,
                max_gbfs_steps,
                runtime_device,
            )
        per_solved_fixed = 100 * float(sum(is_solved_fixed)) / float(len(is_solved_fixed))

        if rank == 0:
            logger.info(f"Greedy policy solved: {per_solved_fixed}\nGreedy policy solved (best): {per_solved_best}")

        if per_solved_fixed > per_solved_best:
            per_solved_best = per_solved_fixed
            update_nnet = True
        else:
            update_nnet = False

        if rank == 0:
            logger.info("Generated test states")

        num_actions = env.num_actions_max
        assert num_actions is not None, "num_actions_max should not be None"

        # Use inference_mode for test evaluation
        with torch.inference_mode():
            per_solved_test = evaluate_greedy_policy(
                states_offline_np,
                dqn_cfg.num_test,
                dqn,
                transition_model,
                num_actions,
                dqn_cfg.goal_steps,
                runtime_device,
                max_gbfs_steps,
                dqn_cfg.per_eq_tol,
            )

        if tracking_session is not None:
            log_metrics(
                tracking_session,
                {
                    "train/last_loss": last_loss,
                    "eval/per_solved_fixed": per_solved_fixed,
                    "eval/per_solved_best": per_solved_best,
                    "eval/per_solved_test": per_solved_test,
                },
                step=itr,
            )

        if rank == 0:
            logger.info(f"Test time: {time.time() - start_time:.2f}")

        if reporter is not None and rank == 0 and callable(reporter):
            reporter({
                "iteration": itr,
                "primary": per_solved_test,
                "metrics": {
                    "last_loss": last_loss,
                    "per_solved_fixed": per_solved_fixed,
                    "per_solved_best": per_solved_best,
                    "per_solved_test": per_solved_test,
                },
            })

        # Release cached CUDA blocks between training iterations.
        torch.cuda.empty_cache()
        if is_distributed:
            distributed_barrier_sync()

        # Update target network if needed
        if rank == 0:
            logger.info(f"Last loss was {last_loss}")

        if update_nnet:
            if rank == 0:
                logger.info("Updating target network")
            copy_files(curr_dir, targ_dir)
            update_num += 1

        # Save status
        if rank == 0:
            with pathlib.Path(f"{curr_dir}/status.pkl").open("wb") as f:
                pickle.dump((itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best), f, protocol=-1)

    # Cleanup distributed training
    if is_distributed:
        distributed_barrier_sync()
        if dist.is_initialized():
            dist.destroy_process_group()

    return {
        "metrics": {
            "iteration": itr,
            "update_num": update_num,
            "last_loss": last_loss,
            "per_solved_fixed": per_solved_fixed,
            "per_solved_best": per_solved_best,
            "per_solved_test": per_solved_test,
        },
        "artifacts": {"model_dir": model_dir, "current_dir": curr_dir, "target_dir": targ_dir},
    }
