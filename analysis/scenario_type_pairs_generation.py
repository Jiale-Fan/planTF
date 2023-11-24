import numpy as np
import matplotlib.pyplot as plt

SCENARIO_MAPPING_IDS = {
		'unknown': 0,
    'accelerating_at_crosswalk': 1,
    'accelerating_at_stop_sign': 2,
    'accelerating_at_stop_sign_no_crosswalk': 3,
    'accelerating_at_traffic_light': 4,
    'accelerating_at_traffic_light_with_lead': 5,
    'accelerating_at_traffic_light_without_lead': 6,
    'behind_bike': 7,
    'behind_long_vehicle': 8,
    'behind_pedestrian_on_driveable': 9,
    'behind_pedestrian_on_pickup_dropoff': 10,
    'changing_lane': 11,
    'changing_lane_to_left': 12,
    'changing_lane_to_right': 13,
    'changing_lane_with_lead': 14,
    'changing_lane_with_trail': 15,
    'crossed_by_bike': 16,
    'crossed_by_vehicle': 17,
    'following_lane_with_lead': 18,
    'following_lane_with_slow_lead': 19,
    'following_lane_without_lead': 20,
    'high_lateral_acceleration': 21,
    'high_magnitude_jerk': 22,
    'high_magnitude_speed': 23,
    'low_magnitude_speed': 24,
    'medium_magnitude_speed': 25,
    'near_barrier_on_driveable': 26,
    'near_construction_zone_sign': 27,
    'near_high_speed_vehicle': 28,
    'near_long_vehicle': 29,
    'near_multiple_bikes': 30,
    'near_multiple_pedestrians': 31,
    'near_multiple_vehicles': 32,
    'near_pedestrian_at_pickup_dropoff': 33,
    'near_pedestrian_on_crosswalk': 34,
    'near_pedestrian_on_crosswalk_with_ego': 35,
    'near_trafficcone_on_driveable': 36,
    'on_all_way_stop_intersection': 37,
    'on_carpark': 38,
    'on_intersection': 39,
    'on_pickup_dropoff': 40,
    'on_stopline_crosswalk': 41,
    'on_stopline_stop_sign': 42,
    'on_stopline_traffic_light': 43,
    'on_traffic_light_intersection': 44,
    'starting_high_speed_turn': 45,
    'starting_left_turn': 46,
    'starting_low_speed_turn': 47,
    'starting_protected_cross_turn': 48,
    'starting_protected_noncross_turn': 49,
    'starting_right_turn': 50,
    'starting_straight_stop_sign_intersection_traversal': 51,
    'starting_straight_traffic_light_intersection_traversal': 52,
    'starting_u_turn': 53,
    'starting_unprotected_cross_turn': 54,
    'starting_unprotected_noncross_turn': 55,
    'stationary': 56,
    'stationary_at_crosswalk': 57,
    'stationary_at_traffic_light_with_lead': 58,
    'stationary_at_traffic_light_without_lead': 59,
    'stationary_in_traffic': 60,
    'stopping_at_crosswalk': 61,
    'stopping_at_stop_sign_no_crosswalk': 62,
    'stopping_at_stop_sign_with_lead': 63,
    'stopping_at_stop_sign_without_lead': 64,
    'stopping_at_traffic_light_with_lead': 65,
    'stopping_at_traffic_light_without_lead': 66,
    'stopping_with_lead': 67,
    'traversing_crosswalk': 68,
    'traversing_intersection': 69,
    'traversing_narrow_lane': 70,
    'traversing_pickup_dropoff': 71,
    'traversing_traffic_light_intersection': 72,
    'waiting_for_pedestrian_to_cross': 73
}

SCENARIO_MAPPING_IDS_REVERSE = {v: k for k, v in SCENARIO_MAPPING_IDS.items()}

keywords_behavior = ['accelerating', 'changing', 'following', 'starting', 'stationary', 'stopping', 'traversing', 'waiting', 'turn']
keywords_environment = ['barrier', 'crosswalk', 'driveable', 'intersection', 'pickup_dropoff', 'stop_sign', 'stopline', 'traffic_light', ]
keywords_object = ['bike', 'pedestrian', 'trafficcone', 'long_vehicle', 'crossed']

def get_scenario_type_pairs(keywords):
    n_types = len(SCENARIO_MAPPING_IDS_REVERSE)
    pair_matrix = -np.ones([n_types, n_types])
    for i in range(n_types):
        pair_matrix[i,i]=1
        kws_present = []
        for kw in keywords:
            if kw in SCENARIO_MAPPING_IDS_REVERSE[i]:
                kws_present.append(kw)
        for j in range(n_types):
            for kw_pre in kws_present:
                if kw_pre in SCENARIO_MAPPING_IDS_REVERSE[j]:
                    pair_matrix[i,j] = 1
                    break
    pair_matrix[0,:] = 0
    pair_matrix[:,0] = 0
    return pair_matrix

behavior_mat = get_scenario_type_pairs(keywords_behavior)
environment_mat = get_scenario_type_pairs(keywords_environment)
object_mat = get_scenario_type_pairs(keywords_object)

# visualize the matrix


# Create a figure with subplots
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# # Plot behavior_mat
# axes[0].imshow(behavior_mat, cmap='binary')
# axes[0].set_title('Behavior Matrix')

# # Plot environment_mat
# axes[1].imshow(environment_mat, cmap='binary')
# axes[1].set_title('Environment Matrix')

# # Plot object_mat
# axes[2].imshow(object_mat, cmap='binary')
# axes[2].set_title('Object Matrix')

# # Adjust spacing between subplots
# plt.tight_layout()

# # Show the plot
# plt.show()
np.set_printoptions(threshold=1e4)
array_code = repr(object_mat)
print(array_code)