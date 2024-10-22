# we intend to create a small training set from the large training set.
# In this way we can train the model faster to see if it is working as expected.

import os
import random
import shutil
from tqdm import tqdm

def copy_random_subfolders(source_dir, destination_dir, fraction=0.1):
    # Get the list of all subfolders in the source directory
    subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
    
    # Calculate the number of subfolders to copy (1/10 of the total)
    num_to_copy = max(1, int(len(subfolders) * fraction))
    
    # Randomly select subfolders to copy
    subfolders_to_copy = random.sample(subfolders, num_to_copy)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Copy the selected subfolders to the destination directory with a progress bar
    for subfolder in tqdm(subfolders_to_copy, desc="Copying Folders", unit="folder"):
        folder_name = os.path.basename(subfolder)
        destination_path = os.path.join(destination_dir, folder_name)
        shutil.copytree(subfolder, destination_path)

if __name__ == "__main__":
    source_directory = "/media/jiale/Jiale_SSD1/cache_1M"  # Replace with your source directory path
    destination_directory = "/media/jiale/T72/cache_100k"  # Replace with your destination directory path
    
    copy_random_subfolders(source_directory, destination_directory, fraction=0.1)  # 1/10th of the subfolders
