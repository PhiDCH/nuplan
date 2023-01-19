import warnings
warnings.filterwarnings("ignore")

from loguru import logger
# Useful imports
import os
from pathlib import Path
import tempfile

os.environ["NUPLAN_DATA_ROOT"] = "/home/robotic/Downloads/nuplan"
os.environ["NUPLAN_MAPS_ROOT"] = '/home/robotic/Downloads/nuplan/dataset/maps'
os.environ["NUPLAN_DB_FILES"] = '/home/robotic/Downloads/nuplan/nuplan-v1.1/mini'
os.environ["NUPLAN_MAP_VERSION"] = 'nuplan-maps-v1.0'
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '~/Downloads/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '~/Downloads/nuplan/dataset/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '~/Downloads/nuplan/nuplan-v1.1/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')

import hydra
from omegaconf import OmegaConf, DictConfig
from typing import List, Type

import numpy as np
import numpy.typing as npt


# Create a temporary directory to store the cache and experiment artifacts
SAVE_DIR = os.path.join('tmp' ,'tutorial_nuplan_framework')  # optionally replace with persistent dir
os.makedirs(SAVE_DIR, exist_ok=True)
EXPERIMENT = 'simulation_simple_experiment'
LOG_DIR = os.path.join(SAVE_DIR, EXPERIMENT)
os.makedirs(LOG_DIR, exist_ok=True)

# Location of path with all simulation configs
CONFIG_PATH = 'nuplan-devkit/nuplan/planning/script/config/simulation'
CONFIG_NAME = 'default_simulation'


MODEL_PATH = "/home/robotic/Downloads/nuplan/tmp/tutorial_nuplan_framework/training_raster_experiment/2023.01.04.15.15.02/best_model/epoch.ckpt"

# Select the planner and simulation challenge
PLANNER = 'ml_planner'  # [simple_planner, ml_planner]
CHALLENGE = 'open_loop_boxes'  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_filter=all_scenarios',  # initially select all scenarios in the database
    # 'scenario_filter=one_continuous_log',  # initially select all scenarios in the database
    'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
    'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
]

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'experiment_name={EXPERIMENT}',
    f'group={SAVE_DIR}',
    'planner=ml_planner',
    'model=raster_model',
    'planner.ml_planner.model_config=${model}',  # hydra notation to select model config
    f'planner.ml_planner.checkpoint_path={MODEL_PATH}',  # this path can be replaced by the checkpoint of the model trained in the previous section
    f'+simulation={CHALLENGE}',
    'output_dir=${group}/${experiment}',
    *DATASET_PARAMS,
    'worker=single_machine_thread_pool',
    # 'worker.use_process_pool=True'
])


from nuplan.planning.script.run_simulation import main as main_simulation

# Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
main_simulation(cfg)

# Simple simulation folder for visualization in nuBoard
simple_simulation_folder = cfg.output_dir
print(simple_simulation_folder)