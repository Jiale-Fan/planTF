import logging
from typing import Optional

import hydra
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from omegaconf import DictConfig

from src.custom_training.custom_training_builder import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


@hydra.main(config_path="./config", config_name="planTF_training")
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == "train":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
        return engine
    if cfg.py_func == "validate":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
        return engine
    elif cfg.py_func == "test":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info("Starting testing...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == "cache":
        # Precompute and cache all features
        logger.info("Starting caching...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()

# command to cache data
'''
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_planTF \
    scenario_builder=nuplan \
    cache.cache_path=/data1/nuplan/jiale2/exp/cache_plantf_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=16
'''

# command to train
'''
export CUDA_VISIBLE_DEVICES=0,1,2
python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/data1/nuplan/jiale2/exp/cache_plantf_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=43 data_loader.params.num_workers=32 \
  data_loader.datamodule.train_fraction=1.0 \
  data_loader.datamodule.val_fraction=0.1 \
  data_loader.datamodule.test_fraction=0.1 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  wandb.mode=online wandb.project=nuplan wandb.name=plantf
  '''  

# wandb.mode=online wandb.project=nuplan wandb.name=plantf

'''
CUDA_VISIBLE_DEVICES=0,1,2 python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/data1/nuplan/jiale2/exp/cache_plantf_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=43 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  wandb.mode=online wandb.project=nuplan wandb.name=plantf
'''