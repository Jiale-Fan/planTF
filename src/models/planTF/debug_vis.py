import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from src.utils.conversion import to_device, to_numpy, to_tensor
import os

def plot_scene_attention(hist_trajs, trajectory_valid_mask, map_segments, attention_weights, key_padding_mask,
                         planned_trajs, filename='1', savepath='./debug_files/', prefix=None):
    """
    Plot the scene with color-coded attention weights.

    Args:
    - map_segments: [M, P, D].
    - trajectories: [N, T, D].
    - trajectory_valid_mask: [N, T].
    - attention_weights: [N+M].
    - planned_trajs: [T, D].

    Returns:
    - None
    """
    cwd = os.getcwd()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if type(filename) != str:
        filename = f'{filename:04d}'
    if filename[-4:] != '.png':
        filename += '.png'

    if prefix is not None:
        filename = str(prefix) +'_'+ filename

    hist_trajs = to_numpy(hist_trajs)
    map_segments = to_numpy(map_segments)
    attention_weights = to_numpy(attention_weights)
    trajectory_valid_mask = to_numpy(trajectory_valid_mask)
    planned_trajs = to_numpy(planned_trajs)
    key_padding_mask = to_numpy(key_padding_mask)

    plt.clf()
    fig, ax = plt.subplots()
    A, T, D1 = hist_trajs.shape
    M, P, D2 = map_segments.shape
    normed_attn_weights = attention_weights / np.max(attention_weights)
    colors = cm.magma(normed_attn_weights)

    # sort the attention weights to make sure that the most important segments are plotted on top
    sorted_idx=np.argsort(attention_weights[A:])

    # plt.grid(True)

    # plot map segments
    for m in range(M):
        idx_m = sorted_idx[m]
        if key_padding_mask[A+idx_m]:
            continue
        segment_data = map_segments[idx_m]
        x = segment_data[:, 0]
        y = segment_data[:, 1]
        plt.scatter(x, y, c=[colors[A+idx_m]], label=f'Segment {idx_m + 1}', s=0.5, marker='o')

    # plot hist trajectories
    # use different markers and color map for trajectories to distinguish them from map segments
    # normed_attn_weights_traj = attention_weights[:A] / np.max(attention_weights[:A])
    normed_attn_weights_traj = normed_attn_weights
    colors_traj = cm.summer(normed_attn_weights_traj) 
    for a in range(A):
        if not np.any(trajectory_valid_mask[a]):
            continue
        segment_data = hist_trajs[a]
        x = segment_data[trajectory_valid_mask[a], 0]
        y = segment_data[trajectory_valid_mask[a], 1]
        plt.scatter(x, y, c=[colors_traj[a]], label=f'Traj {a + 1}', s=0.2, marker='x')

    # plot planned trajectories
    plt.scatter(hist_trajs[0,-1,0], hist_trajs[0,-1,1], c='r', label=f'Ego Position', s=6, marker='*')
    plt.scatter(planned_trajs[:, 0], planned_trajs[:, 1], c='c', label=f'Planned Trajectory', s=0.5, marker='o')

    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.magma), ax=ax)
    cbar.set_label('Lane Attention')
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.summer), ax=ax)
    cbar.set_label('Agent Attention')

    plt.axis('equal')
    plt.title('Scatter Plot of 3D Tensor')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()
    
    plt.savefig(savepath + filename, dpi=600)
    # close the fig
    plt.close(fig)

def plot_lane_segments(tensor, filename='1.png', savepath='/home/jiale/planTF/debug_files/', color=None):
    """
    Plot 3D tensor data using scatter plot.

    Args:
    - tensor (numpy.ndarray): 3D tensor of shape [M, P, D].

    Returns:
    - None
    """
    plt.clf()
    M, P, D = tensor.shape
    colors = plt.cm.prism(np.linspace(0, 1, M))  # Generate M different colors

    for m in range(M):
        segment_data = tensor[m]
        x = segment_data[:, 0]
        y = segment_data[:, 1]
        
        if color == None:
            plt.scatter(x, y, c=[colors[m]], label=f'Segment {m + 1}', s=0.5)
        else:
            plt.scatter(x, y, c=[color], label=f'Segment {m + 1}', s=0.5)

    plt.title('Scatter Plot of 3D Tensor')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()
    plt.grid(True)
    plt.savefig(savepath + filename, dpi=300)