# read the pickle file
import pickle
import torch
import matplotlib.pyplot as plt



def visualize_tree(dictionary, indent=0):
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            print(' ' * indent + str(key) + ': ' + str(value.shape))
        else:
            print(' ' * indent + str(key))
        if isinstance(value, dict):
            visualize_tree(value, indent + 2)
        

# visualize_tree(data)

'''
agent
  position: torch.Size([2, 33, 101, 2])
  heading: torch.Size([2, 33, 101])
  velocity: torch.Size([2, 33, 101, 2])
  shape: torch.Size([2, 33, 101, 2])
  category: torch.Size([2, 33])
  valid_mask: torch.Size([2, 33, 101])
  target: torch.Size([2, 33, 80, 3])
map
  point_position: torch.Size([2, 95, 3, 20, 2])
  point_vector: torch.Size([2, 95, 3, 20, 2])
  point_orientation: torch.Size([2, 95, 3, 20])
  point_side: torch.Size([2, 95, 3]) # ? [1,2,3] always
  polygon_center: torch.Size([2, 95, 3])
  polygon_position: torch.Size([2, 95, 2])
  polygon_orientation: torch.Size([2, 95])
  polygon_type: torch.Size([2, 95])
  polygon_on_route: torch.Size([2, 95])
  polygon_tl_status: torch.Size([2, 95]) # traffic light status
  polygon_has_speed_limit: torch.Size([2, 95])
  polygon_speed_limit: torch.Size([2, 95])
  valid_mask: torch.Size([2, 95, 20])
current_state: torch.Size([2, 7])
origin: torch.Size([2, 2])
angle: torch.Size([2])
scenario_type: torch.Size([2, 1])
'''


def plot_sample_elements(data, sample_index, out):

    map = data['map']
    target = data['agent']['target'][sample_index, 0]

    planned_trajectories = out['trajectory'][sample_index, :, :, :2].detach().cpu().numpy()
    
    # point_vector = data['point_vector'][sample_index]
    # point_orientation = data['point_orientation'][sample_index]
    # point_side = data['point_side'][sample_index]
    # polygon_center = data['polygon_center'][sample_index]
    
    # polygon_orientation = data['polygon_orientation'][sample_index]
    # polygon_type = data['polygon_type'][sample_index]
    
    # polygon_tl_status = data['polygon_tl_status'][sample_index]
    # polygon_has_speed_limit = data['polygon_has_speed_limit'][sample_index]
    # polygon_speed_limit = data['polygon_speed_limit'][sample_index]

    valid_mask = map['valid_mask'][sample_index]
    target = target.to('cpu').numpy()
    polygon_on_route = map['polygon_on_route'][sample_index].to('cpu').numpy()
    polygon_position = map['polygon_position'][sample_index].to('cpu').numpy()
    point_position = map['point_position'][sample_index].to('cpu').numpy() # points position of lane object and crosswalk object

    # Plotting code for each element
    plt.figure(figsize=(10, 6))
    # Example: Plotting point_position
    plt.scatter(point_position[:, 0, :, 0].flatten(), point_position[:, 0, :, 1].flatten(), c='black', s=2)
    # plt.legend(['Point Position'])
    # plt.scatter(point_position[:, 1, :, 0].flatten(), point_position[:, 1, :, 1].flatten(), c='r')
    # plt.scatter(point_position[:, 2, :, 0].flatten(), point_position[:, 2, :, 1].flatten(), c='g')

    plt.scatter(polygon_position[:, 0], polygon_position[:, 1], c='k', s=0.5)

    plt.scatter(point_position[polygon_on_route!=0, 0, :, 0].flatten(), point_position[polygon_on_route!=0, 0, :, 1].flatten(), c='r', s=2)
    

    # plot the planned trajectories
    for j in range(planned_trajectories.shape[0]):
        plt.plot(planned_trajectories[j, :, 0], planned_trajectories[j, :, 1], c='g', linewidth=2)
    plt.plot(target[:, 0], target[:, 1], c='b', linewidth=3)

    # equal axis
    plt.axis('equal')

    plt.legend(['Polygon Position'])
    plt.xlabel('X')
    plt.ylabel('Y')

    # Add more plotting code for other elements...

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # read the pickle file
    with open('./debug_files/nuplan_feature_data.pkl', 'rb') as f:
        data = pickle.load(f)

    pass
   