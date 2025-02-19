import pandas as pd
import yaml

# Read the parquet file
parquet_file = 'val14/aggregator_metric/test14_random.parquet'
df = pd.read_parquet(parquet_file)

# Extract the "scenario" column
scenarios = df['scenario'].tolist()

# Create a dictionary to hold the data
data = {'scenario': scenarios}

# Custom representer to ensure double quotes
def str_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

yaml.add_representer(str, str_presenter)

# Write the data to a YAML file
yaml_file = 'scenario_list.yaml'
with open(yaml_file, 'w') as file:
    yaml.dump(data, file, default_flow_style=False)