import torch

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from typing import List, Tuple
import matplotlib.pyplot as plt

class FasterMetricsComputer:
    def __init__(self):
        pass

        # NOTE: if running time is too long, we can downsample the trajectories

    def _coordinates_transform(self, pos, heading, points):
        """
        Transform coordinates of points from the local frame to the global frame
        Itorchut:
            pos: tensor or numpy array of shape (2,)
            heading: scalar
            points: tensor or numpy array of shape (..., 2)
        Output:
            points_global: tensor or numpy array of shape (..., 2)
        """

        # rotation matrix should be the transpose of the rotation matrix in nuplan feature normalization
        rotation_mat = torch.tensor([[torch.cos(heading), torch.sin(heading)],
                            [-torch.sin(heading), torch.cos(heading)]])
        points_global = torch.matmul(points, rotation_mat.to(points.device)) + pos
        return points_global
    
    def _coordinates_transform_batch(self, origins, angles, points):

        # rotation matrix should be the transpose of the rotation matrix in nuplan feature normalization
        batch_traj_list = [self._coordinates_transform(origins[i], angles[i], points[i]) for i in range(origins.shape[0])]
        points_global = torch.stack(batch_traj_list, dim=0)
        return points_global

    
    # def _coordinates_transform_batch(self, origins, angles, points):

    #     # rotation matrix should be the transpose of the rotation matrix in nuplan feature normalization
    #     rotation_mat = torch.stack([torch.cos(angles), torch.sin(angles),
    #                         -torch.sin(angles), torch.cos(angles)], dim=-1).squeeze().view(-1, 2, 2)
    #     points_global = torch.bmm(points, rotation_mat) + origins[:, None, :]
    #     return points_global

    
    def calculate_metric_drivable_area_batch(self, scenarios: List[NuPlanScenario], trajectories, origins, angles):
        """
        Calculate the drivable area metric for a given scenario and trajectories
        Input: 
            scenario: NuPlanScenario
            trajectories: tensor or numpy array of shape (bs, num_traj, num_timesteps, 3)
        """

        # TODO: we can have the corners or other interested points of the car as points
        # we want to query the drivable area at

        # trajectories coordinates transformation
        trajectories_global = self._coordinates_transform_batch(origins, angles, trajectories[..., :2]) # (num_traj, num_timesteps, 2)
        # trajectories_global [bs, num_trajs, num_timesteps, 2]

        # for each trajectory, query that if each waypoint is in the drivable area
        
        batch_score_array = []
        for j in range(trajectories_global.shape[0]):
            traj_score = []
            for i in range(trajectories_global.shape[1]):
                traj = trajectories_global[j, i].cpu()
                points_2d_list = [Point2D(x=traj[k, 0], y=traj[k, 1]) for k in range(traj.shape[0])]
                res = torch.tensor([scenarios[j].map_api.is_in_layer(point, SemanticMapLayer.DRIVABLE_AREA) for point in points_2d_list])
                satisfied = torch.all(res)
                traj_score.append(satisfied)

            traj_score_array = torch.tensor(traj_score)
            batch_score_array.append(traj_score_array)

        batch_score_array = torch.stack(batch_score_array, dim=0)

        return batch_score_array
    
    def _sanity_check(self, scenarios: List[NuPlanScenario], trajectories, origins, angles):
        """
        Check if the ground truth of the trajectories satisfy the constraints of the scenarios
        Input:
            scenarios: list of NuPlanScenario
            trajectories: tensor or numpy array of shape (num_scenarios, num_timesteps, 3)
        """
        assert len(scenarios) == trajectories.shape[0]

        traj_score = []
        for i in range(len(scenarios)):
            scenario = scenarios[i]
            traj = trajectories[i]

            traj_global = self._coordinates_transform(origins[i], angles[i], traj[...,:2]).cpu() # (num_traj, num_timesteps, 2)

            points_2d_list = [Point2D(x=traj_global[j, 0], y=traj_global[j, 1]) for j in range(traj_global.shape[0])]
            res = torch.tensor([scenario.map_api.is_in_layer(point, SemanticMapLayer.DRIVABLE_AREA) for point in points_2d_list])
            satisfied = torch.all(res)
            traj_score.append(satisfied)

            if not satisfied:
                pass
                self._plot_map_trajectory(scenario.map_api, traj_global)
    
    def _plot_map_trajectory(self, map_api, trajectory):
        """
        Plot a trajectory on the map
        Input:
            map_api: NuplanMap
            trajectory: tensor or numpy array of shape (num_timesteps, 3)
        """
        
        trajectory = trajectory.cpu().numpy()
        map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA).plot()
        plt.plot(trajectory[:,0], trajectory[:,1], 'r')
        plt.scatter(trajectory[0,0], trajectory[0,1], marker='*', c='r')
        plt.xlim(trajectory[0,0]-100, trajectory[0,0]+100)
        plt.ylim(trajectory[0,1]-100, trajectory[0,1]+100)

    def _plot_trajectories(self, map_api, trajectories):
        """
        Plot a list of trajectories on the map
        Input:
            map_api: NuplanMap
            trajectories: tensor or numpy array of shape (num_traj, num_timesteps, 3)
        """

        trajectory = trajectory.cpu().numpy()
        map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA).plot()
        for traj in trajectories:
            plt.plot(traj[:,0], traj[:,1])