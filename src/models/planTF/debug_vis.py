import matplotlib.pyplot as plt
import numpy as np


def plot_scene_points(tensor, filename, savepath='/home/jiale/planTF/debug_files/'):
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
        
        plt.scatter(x, y, c=[colors[m]], label=f'Segment {m + 1}', s=0.5)

    plt.title('Scatter Plot of 3D Tensor')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()
    plt.grid(True)
    plt.savefig(savepath + filename, dpi=300)