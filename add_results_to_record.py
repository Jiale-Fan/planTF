import pandas as pd
import os

def add_results_to_record(input_parquet_path, log_parquet_path, sheet_name):
    # copy the input parquet file to the results folder
    import shutil
    shutil.copy(input_parquet_path, os.path.expanduser('~/Documents/results/' + sheet_name + '.parquet'))
    # Read the input parquet file
    new_data = pd.read_parquet(input_parquet_path)
    
    # Check if the log parquet file exists
    if os.path.exists(log_parquet_path):
        # Read the existing log parquet file
        log_data = pd.read_parquet(log_parquet_path)
    else:
        # Create an empty DataFrame if the log file does not exist
        log_data = pd.DataFrame(columns=new_data.columns)
        # take the row "score" from the new_data and add it to the log_data
    log_data = pd.concat([log_data, new_data.loc[new_data['scenario'] == 'final_score']], ignore_index=True)
    # Write the log parquet file
    log_data.to_parquet(log_parquet_path)

# Example usage
exp_name = 'best_1.2_ips0'
val_set_name = 'test14-random' # 'test14-random', 'test14-hard', 'val14'
for sim_type in ['open_loop_boxes', 'closed_loop_nonreactive_agents', 'closed_loop_reactive_agents']:
    input_parquet_path = '/home/jiale/Documents/exp/exp/simulation/' + sim_type + '/' + val_set_name + '/aggregator_metric/'
    # read the only file in the directory
    input_parquet_path = input_parquet_path + os.listdir(input_parquet_path)[0]
    log_parquet_path = './all_results.parquet'
    sheet_name = exp_name + '--' + sim_type + '--' + val_set_name
    add_results_to_record(input_parquet_path, log_parquet_path, sheet_name)

pass