"""Register SPAR's structured Pydantic schemas with Hydra ConfigStore."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from . import config_schema as cfg_schema


def register_configs() -> None:
    """Register the structured configuration schemas with Hydra ConfigStore."""
    cs: ConfigStore = ConfigStore.instance()

    # Register base configurations
    cs.store(group="wandb", name="base", node=cfg_schema.WandbConfig)
    cs.store(group="env", name="base", node=cfg_schema.EnvConfig)
    cs.store(group="model", name="base", node=cfg_schema.ModelConfig)
    cs.store(group="data", name="base", node=cfg_schema.DataConfig)
    cs.store(group="search_data", name="base", node=cfg_schema.SearchPairsDataConfig)
    cs.store(group="train", name="base", node=cfg_schema.TrainConfig)
    cs.store(group="test", name="base", node=cfg_schema.TestConfig)
    cs.store(group="search", name="base", node=cfg_schema.SearchConfig)
    cs.store(group="visualization", name="base", node=cfg_schema.VisualizationConfig)
    cs.store(group="plotter", name="base", node=cfg_schema.PlotterConfig)
    cs.store(group="optuna", name="base", node=cfg_schema.OptunaConfig)

    # Register nested configurations
    cs.store(group="data/dataset", name="base", node=cfg_schema.DatasetConfig)
    cs.store(group="search_data/dataset", name="base", node=cfg_schema.SearchPairsDatasetConfig)
    cs.store(group="data/train_data_path", name="base", node=cfg_schema.TrainDataPathConfig)
    cs.store(group="train/dataloader", name="base", node=cfg_schema.DataLoaderConfig)
    cs.store(group="test/dataloader", name="base", node=cfg_schema.TestDataLoaderConfig)
    cs.store(group="train/scheduler", name="base", node=cfg_schema.SchedulerConfig)
    cs.store(group="train/phase", name="base", node=cfg_schema.TrainPhaseConfig)
    cs.store(group="model/alignment_model", name="base", node=cfg_schema.AlignmentModelConfig)
    cs.store(group="model/eval_model", name="base", node=cfg_schema.EvalModelConfig)
    cs.store(group="model/encoder", name="base", node=cfg_schema.EncoderConfig)
    cs.store(group="model/decoder", name="base", node=cfg_schema.DecoderConfig)
    cs.store(group="model/env_model", name="base", node=cfg_schema.EnvModelConfig)
    cs.store(group="sweep", name="base", node=cfg_schema.SweepConfig)
    cs.store(group="dataloader", name="base", node=cfg_schema.DataLoaderConfig)
    cs.store(group="optuna/study", name="base", node=cfg_schema.StudyConfig)
    cs.store(group="optuna/storage", name="base", node=cfg_schema.StorageConfig)
    cs.store(group="optuna/sampler", name="base", node=cfg_schema.SamplerConfig)
    cs.store(group="optuna/pruner", name="base", node=cfg_schema.PrunerConfig)
    cs.store(group="optuna/runtime", name="base", node=cfg_schema.OptunaRuntimeConfig)
    cs.store(group="optuna/analysis", name="base", node=cfg_schema.AnalysisConfig)
    cs.store(group="optuna/replay", name="base", node=cfg_schema.ReplayConfig)

    # Register stage-specific SPAR configurations
    cs.store(group="schema", name="base_schema", node=cfg_schema.BaseSPARConfig)
    cs.store(group="schema", name="gen_data_schema", node=cfg_schema.GenDataSPARConfig)
    cs.store(group="schema", name="gen_search_data_schema", node=cfg_schema.GenSearchDataSPARConfig)
    cs.store(group="schema", name="create_sweep_schema", node=cfg_schema.CreateSweepSPARConfig)
    cs.store(group="schema", name="encode_offline_data_schema", node=cfg_schema.EncodeOfflineDataSPARConfig)
    cs.store(group="schema", name="train_world_model_schema", node=cfg_schema.TrainEnvModelSPARConfig)
    cs.store(group="schema", name="train_env_disc_schema", node=cfg_schema.TrainEnvDiscSPARConfig)
    cs.store(group="schema", name="train_env_cont_schema", node=cfg_schema.TrainEnvContSPARConfig)
    cs.store(group="schema", name="train_alignment_model_schema", node=cfg_schema.TrainAlignmentModelSPARConfig)
    cs.store(group="schema", name="train_alignment_disc_schema", node=cfg_schema.TrainAlignmentDiscSPARConfig)
    cs.store(group="schema", name="train_alignment_cont_schema", node=cfg_schema.TrainAlignmentContSPARConfig)
    cs.store(group="schema", name="train_heuristic_schema", node=cfg_schema.TrainHeuristicSPARConfig)
    cs.store(group="schema", name="search_gbfs_schema", node=cfg_schema.SearchGBFSSPARConfig)
    cs.store(group="schema", name="search_qstar_schema", node=cfg_schema.SearchQStarSPARConfig)
    cs.store(group="schema", name="visualize_unsolved_qstar_schema", node=cfg_schema.VisualizeUnsolvedQStarSPARConfig)
    cs.store(group="schema", name="bitwise_eq_report_schema", node=cfg_schema.BitwiseEqReportSPARConfig)
    cs.store(group="schema", name="qstar_results_to_latex_schema", node=cfg_schema.QStarResultsToLatexSPARConfig)
    cs.store(group="schema", name="search_ucs_schema", node=cfg_schema.SearchUCSSPARConfig)
    cs.store(group="schema", name="plotter_schema", node=cfg_schema.PlotterSPARConfig)
    cs.store(group="schema", name="mse_plotter_schema", node=cfg_schema.MSEPlotterSPARConfig)
    cs.store(group="schema", name="test_model_schema", node=cfg_schema.TestModelSPARConfig)
    cs.store(group="schema", name="test_model_disc_schema", node=cfg_schema.TestModelDiscSPARConfig)
    cs.store(group="schema", name="test_model_cont_schema", node=cfg_schema.TestModelContSPARConfig)
    cs.store(group="schema", name="process_image_schema", node=cfg_schema.ProcessImageSPARConfig)
    cs.store(group="schema", name="optuna_study_schema", node=cfg_schema.OptunaStudySPARConfig)
    cs.store(group="schema", name="optuna_analyze_schema", node=cfg_schema.OptunaAnalyzeSPARConfig)
    cs.store(group="schema", name="optuna_replay_schema", node=cfg_schema.OptunaReplaySPARConfig)
    cs.store(
        group="schema",
        name="alignment_encoder_match_report_schema",
        node=cfg_schema.AlignmentEncoderMatchReportSPARConfig,
    )


# Importing this module registers the configuration groups below.
register_configs()
