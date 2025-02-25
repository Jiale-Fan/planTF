import os
import yaml

def check_files_exist(yaml_path, trainval_dir, test_directory, split='val'):
    """
    Check if all the log files of that split exist in the given directory.

    """

    assert split in ['train', 'val', 'test'], "split must be one of 'train', 'val', or 'test'."

    data_dir = trainval_dir if split in ['train', 'val'] else test_directory

    # Load the YAML file
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Get the list of files in the 'val' split
    complete_files_list = data.get('log_splits', {}).get(split, [])
    
    # Check if each file exists in the given directory
    missing_files_list = []
    for file_name in complete_files_list:
        file_path = os.path.join(data_dir, file_name+'.db')
        if not os.path.exists(file_path):
            missing_files_list.append(file_name)
    
    if missing_files_list:
        print("The following files are missing:")
        for file_name in missing_files_list:
            print(file_name)

        n_ms = len(missing_files_list)
        n_cs = len(complete_files_list)
        print("Missing files: {}/{}".format(n_ms, n_cs))
        print("Missing percentage: {:.2f}%".format(n_ms/n_cs * 100))
    else:
        print("All files "+ str(len(complete_files_list))+" in the " +split+ " split exist in the directory.")

# Example usage
yaml_path = '/home/jiale/nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml'
trainval_dir = '/data1/nuplan/dataset/nuplan-v1.1/trainval/splits/trainval'
test_dir = "/media/jiale/T72/test_dataset/data/cache/test/"
check_files_exist(yaml_path, trainval_dir, test_dir, split='val')