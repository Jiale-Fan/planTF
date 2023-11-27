import pickle
import numpy as np

from src.features.nuplan_feature import NuplanFeature
from visualization import plot_sample_elements
import matplotlib.pyplot as plt




def test_original_feature_class():
    # read the pickle file
    with open('./debug_files/nuplan_feature_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # before
    ori_obj = NuplanFeature(data=data)
    single_ori_obj = ori_obj.unpack()[0]
    plot_sample_elements(single_ori_obj.to_numpy().data)

    print(single_ori_obj.data['current_state'][2])
    
    # after
    temp_obj=single_ori_obj.to_numpy()
    temp_obj.data['current_state'][2] += np.pi/4
    print(single_ori_obj.data['current_state'][2])

    normed_obj=NuplanFeature.normalize(temp_obj.data)
    plot_sample_elements(normed_obj.data)

def test_updated_feature_class():
    # read the pickle file
    with open('./debug_files/nuplan_feature_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # before
    ori_obj = NuplanFeature(data=data)
    single_ori_obj = ori_obj.unpack()[0]
    single_ori_obj=single_ori_obj.to_numpy()

    vis_raster_map(single_ori_obj.data['drivable_raster']['data'], )
    
    plot_sample_elements(single_ori_obj.data)

    print(single_ori_obj.data['current_state'][2])
    
    # after
    temp_obj=single_ori_obj.to_numpy()
    temp_obj.data['current_state'][2] += np.pi/4
    print(single_ori_obj.data['current_state'][2])

    normed_obj=NuplanFeature.normalize(temp_obj.data)
    plot_sample_elements(normed_obj.data)

def vis_raster_map(raster, query_point=None):
    
    # visualize raster map using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(raster, cmap='gray')

    
if __name__ == "__main__":
    
    # test_original_feature_class()
    test_updated_feature_class()