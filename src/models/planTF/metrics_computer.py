from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# new imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)

from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from typing import Any, List, Optional, Type, Dict
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
import torch

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
)
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Polygon
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)

count_no_route = 0


class MetricsComputerObservation(PDMObservation):
    def __init__(self, trajectory_sampling: TrajectorySampling, proposal_sampling: TrajectorySampling, map_radius: float, observation_sample_res: int = 2):
        super().__init__(trajectory_sampling, proposal_sampling, map_radius, observation_sample_res)

    def update(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps: List[PDMOccupancyMap] = []
        self._object_manager = self._get_object_manager(ego_state, observation)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = self._object_manager.get_nearest_objects(ego_state.center.point)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_coords_t = (
                    dynamic_object_coords + delta_t * dynamic_object_dxy[:, None]
                )
                dynamic_object_polygons = shapely.creation.polygons(
                    dynamic_object_coords_t
                )

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
                all_polygons,
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(
                    self._occupancy_maps[0][intersecting_obstacle]
                )
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._initialized = True


class MetricsSupervisor(AbstractPDMPlanner):
    """
    Metrics computer modified on the basis of tuplan-garage's AbstractPDMClosedPlanner
    """

    def __init__(
        self,
        map_radius: float = 60.0,
    ):
        """
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(MetricsSupervisor, self).__init__(map_radius)
        self.proposal_sampling = TrajectorySampling(num_poses=80, interval_length=0.1) # TODO check num poses
        self._scorer = PDMScorer(self.proposal_sampling)
        self._simulator = PDMSimulator(self.proposal_sampling)

    def compute_metrics_batch(self, trajectories: torch.Tensor, scenarios: List[NuPlanScenario]) -> Union[np.float64, torch.Tensor]:
        '''
        Args:
            trajectories: [batch, num_modes, time_steps, 6]
            scenarios: list of NuPlanScenario
        '''
        trajectories_3d = torch.concat([trajectories[..., 0:2], trajectories[..., -1:]], dim=-1)

        trajectory_list = np.split(trajectories_3d, trajectories.shape[0])
        metrics_list = [self.compute_metrics(trajectory.squeeze(0), scenario) for trajectory, scenario in zip(trajectory_list, scenarios)]
        mean_metrics = np.mean(metrics_list)

        return mean_metrics, torch.Tensor(metrics_list)

    def compute_metrics(self, trajectory: torch.Tensor, scenario: NuPlanScenario) -> np.float64:
        '''
        This function computes metrics for multiple modes of trajectories for the same scenario
        Args:
            trajectory: [num_modes, time_steps, 3]
            scenario: NuPlanScenario
        Returns:
            metrics: [num_modes]

        '''
        self._map_api = scenario.map_api
        
        # prepare ego states and observation 
        simulation_history_buffer_duration = 2
        buffer_size = int(simulation_history_buffer_duration / scenario.database_interval + 1)

        history = SimulationHistoryBuffer.initialize_from_scenario(
                buffer_size=buffer_size, 
                scenario=scenario, 
                observation_type=DetectionsTracks) 

        ego_state, observation = history.current_state
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # prepare route dicts
        self._load_route_dicts(scenario._route_roadblock_ids)
        if len(self._route_lane_dict) == 0:
            global count_no_route
            count_no_route += 1
            print("No route lane dict found: " + str(count_no_route))
            return np.array([0.5]*trajectory.shape[0])

        # prepare observation
        trajectory_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
        _observation = PDMObservation(
                trajectory_sampling, self.proposal_sampling, self._map_radius
            )

        # update observation
        _observation.update(
            ego_state,
            observation,
            list(scenario.get_traffic_light_status_at_iteration(0)),
            self._route_lane_dict,
        )

        # prepare centerline
        current_lane = self._get_starting_lane(ego_state)
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        _centerline = PDMPath(centerline_discrete_path)

        # get simulated proposal
        proposals_array = np.zeros((trajectory.shape[0], trajectory.shape[1]+1, 11))
        proposals_array[:, 1:, :3] = coordinates_transform(trajectory.cpu().detach().numpy(), ego_state)
        # proposal must be in universal coordinates
        proposals_array[:, 0, :3] = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])

        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        ) # [15, 41, 11]

        # TODO DEBUG tracking performance

        proposal_scores = self._scorer.score_proposals(
                    simulated_proposals_array, # TODO: check dimensions
                    ego_state,
                    _observation,
                    _centerline,
                    self._route_lane_dict,
                    self._drivable_area_map,
                    scenario.map_api,
                    )

        return proposal_scores



    # fake methods
    def name(self) -> str:
        """
        :return string describing name of this planner.
        """
        pass

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Initialize planner
        :param initialization: Initialization class.
        """
        pass


    def observation_type(self) -> Type[Observation]:
        """
        :return Type of observation that is expected in compute_trajectory.
        """
        pass

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: List of planner inputs for which trajectory needs to be computed.
        :return: Trajectories representing the predicted ego's position in future
        """
        pass

def coordinates_transform(proposal: npt.NDArray, ego_state: EgoState):
    '''
    Transform proposal from ego coordinates to universal coordinates
    '''
    ego_state_array = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
    theta = ego_state.rear_axle.heading
    rotation_mat = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
    new_proposal = np.zeros(proposal.shape)
    new_proposal[..., :2] = np.matmul(proposal[..., :2], rotation_mat)
    new_proposal[..., :2] += ego_state_array[:2]
    new_proposal[..., 2] = proposal[..., 2] + theta
    return new_proposal
