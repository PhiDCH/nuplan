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

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters, VehicleParameters
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import  InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel


class SimplePlanner(AbstractPlanner):
    """
    Planner going straight
    """

    def __init__(self,
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32],
                 max_velocity: float = 5.0,
                 steering_angle: float = 0.0):
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        pass

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """ Inherited, see superclass. """
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        # Extract iteration and history
        iteration = current_input[0].iteration
        history = current_input[0].history

        ego_state = history.ego_states[-1]
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                self.acceleration,
            ),
            tire_steering_angle=self.steering_angle,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for _ in np.arange(
            iteration.time_us + self.sampling_time.time_us,
            iteration.time_us + self.horizon_seconds.time_us,
            self.sampling_time.time_us,
        ):  
            # decrease speed when speed > max speed 
            if state.dynamic_car_state.speed > self.max_velocity:
                accel = self.max_velocity - state.dynamic_car_state.speed
                state = EgoState.build_from_rear_axle(
                    rear_axle_pose=state.rear_axle,
                    rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
                    rear_axle_acceleration_2d=StateVector2D(accel, 0),
                    tire_steering_angle=state.tire_steering_angle,
                    time_point=state.time_point,
                    vehicle_parameters=state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=state.dynamic_car_state.angular_velocity,
                    angular_accel=state.dynamic_car_state.angular_acceleration,
                )

            state = self.motion_model.propagate_state(state, state.dynamic_car_state, self.sampling_time)
            trajectory.append(state)

        return [InterpolatedTrajectory(trajectory)]


from dataclasses import dataclass
@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def construct_simulation_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return Hydra config path.
    """
    common_dir = "file://" + os.path.join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = os.path.join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + os.path.join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


# Location of paths with all simulation configs
BASE_CONFIG_PATH = os.path.join(os.getenv('NUPLAN_TUTORIAL_PATH', ''), 'nuplan-devkit/nuplan/planning/script')
simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)


# Create a temporary directory to store the simulation artifacts
# SAVE_DIR = tempfile.mkdtemp()
SAVE_DIR = 'tmp'

# Select simulation parameters
# EGO_CONTROLLER = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
EGO_CONTROLLER = 'log_play_back_controller'  # [log_play_back_controller, perfect_tracking_controller]
OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]
DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database (2.5h of 8 autolabeled logs in Las Vegas)
    'scenario_filter=one_continuous_log',  # simulate only one log
    "scenario_filter.log_names=['2021.05.12.22.00.38_veh-35_01008_01518']",
    'scenario_filter.limit_total_scenarios=2',  # use 2 total scenarios
]

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=simulation_hydra_paths.config_path)

# Compose the configuration
cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=[
    f'group={SAVE_DIR}',
    f'experiment_name=planner_tutorial',
    'worker=sequential',
    f'ego_controller={EGO_CONTROLLER}',
    f'observation={OBSERVATION}',
    f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
    'output_dir=${group}/${experiment}',
    *DATASET_PARAMS,
])

# print(cfg.keys())
# print('train size ', len(cfg.splitter.log_splits.train))
# print('val size ', len(cfg.splitter.log_splits.val))
# print('test size ', len(cfg.splitter.log_splits.test))
# cfg.scenario_builder.scenario_mapping.scenario_map = 1
# cfg.splitter = 1
# logger.info(OmegaConf.to_yaml(cfg))


from nuplan.planning.script.run_simulation import run_simulation as main_simulation

planner = SimplePlanner(horizon_seconds=10.0, sampling_time=0.25, acceleration=[0.0, 0.0])

# Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
main_simulation(cfg, planner)

# Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
# results_dir = list(list(Path(SAVE_DIR).iterdir())[0].iterdir())[0]  # get the child dir 2 levels in
# simulation_file = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]