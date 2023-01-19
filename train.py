from loguru import logger
from omegaconf import OmegaConf

# import warnings
# warnings.filterwarnings("ignore")

# Useful imports
import os
from pathlib import Path
import tempfile

import hydra

os.environ["NUPLAN_DATA_ROOT"] = "/home/robotic/Downloads/nuplan"
os.environ["NUPLAN_MAPS_ROOT"] = '/home/robotic/Downloads/nuplan/dataset/maps'
os.environ["NUPLAN_DB_FILES"] = '/home/robotic/Downloads/nuplan/nuplan-v1.1/mini'
os.environ["NUPLAN_MAP_VERSION"] = 'nuplan-maps-v1.0'
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '~/Downloads/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '~/Downloads/nuplan/dataset/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '~/Downloads/nuplan/nuplan-v1.1/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')


# Location of path with all training configs
CONFIG_PATH = 'nuplan-devkit/nuplan/planning/script/config/training'
CONFIG_NAME = 'default_training'

# Create a temporary directory to store the cache and experiment artifacts
SAVE_DIR = os.path.join('tmp' ,'tutorial_nuplan_framework')  # optionally replace with persistent dir
os.makedirs(SAVE_DIR, exist_ok=True)
EXPERIMENT = 'training_raster_experiment'
LOG_DIR = os.path.join(SAVE_DIR, EXPERIMENT)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration 
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'group={str(SAVE_DIR)}',
    f'cache.cache_path={str(SAVE_DIR)}/cache',
    f'experiment_name={EXPERIMENT}',
    'py_func=cache',
    '+training=training_raster_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_filter.limit_total_scenarios=500',  # Choose 500 scenarios to train with
    'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
    'lightning.trainer.params.max_epochs=30',
    'data_loader.params.batch_size=32',
    'data_loader.params.num_workers=8',

    'cache.cleanup_cache=True', 

    'worker=single_machine_thread_pool'
])


# cfg.splitter = 1
# cfg.scenario_builder.scenario_mapping.scenario_map = 1
# logger.info(OmegaConf.to_yaml(cfg))

# from nuplan.planning.script.run_training import main as main_train

# main_train(cfg)


from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.worker_pool_builder import build_worker

update_config_for_training(cfg)
# print(cfg.worker)
# build_training_experiment_folder(cfg)

worker = build_worker(cfg)
# print(worker.number_of_threads)

from nuplan.planning.training.experiments.training import build_training_engine

engine = build_training_engine(cfg, worker)
print(type(engine.datamodule._train_set))
