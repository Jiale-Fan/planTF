import numpy as np

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

class FasterMetricsComputer:
    def __init__(self):
        pass

        # NOTE: if running time is too long, we can downsample the trajectories

    def coordinates_transform(self, pos, heading, points):
        """
        Transform coordinates of points from the local frame to the global frame
        Input:
            pos: tensor or numpy array of shape (2,)
            heading: scalar
            points: tensor or numpy array of shape (..., 2)
        Output:
            points_global: tensor or numpy array of shape (..., 2)
        """

        # rotation matrix should be the transpose of the rotation matrix in nuplan feature normalization
        rotation_mat = np.array([[np.cos(heading), np.sin(heading)],
                            [-np.sin(heading), np.cos(heading)]], dtype=np.float64)
        points_global = np.matmul(points, rotation_mat) + pos
        return points_global

    
    def calculate_metric_drivable_area(self, scenario: NuPlanScenario, trajectories):
        """
        Calculate the drivable area metric for a given scenario and trajectories
        Input: 
            scenario: NuPlanScenario
            trajectories: tensor or numpy array of shape (num_traj, num_timesteps, 3)
        """

        trajectories = trajectories.cpu().numpy() # (num_traj, num_timesteps, 3)

        # get the initial state of the scenario
        init_pos = scenario.initial_ego_state.rear_axle.array # (2,) 
        init_heading = scenario.initial_ego_state.rear_axle.heading # scalar

        # TODO: we can have the corners or other interested points of the car as points
        # we want to query the drivable area at

        # trajectories coordinates transformation
        trajectories_global = self.coordinates_transform(init_pos, init_heading, trajectories[...,:2]) # (num_traj, num_timesteps, 2)

        # for each trajectory, query that if each waypoint is in the drivable area
        traj_score = []
        for i in range(trajectories_global.shape[0]):
            traj = trajectories_global[i]
            points_2d_list = [Point2D(x=traj[j, 0], y=traj[j, 1]) for j in range(traj.shape[0])]
            res = np.array([scenario.map_api.is_in_layer(point, SemanticMapLayer.DRIVABLE_AREA) for point in points_2d_list])
            traj_score.append(np.prod(res))

        score_array = np.array(traj_score)

        return score_array