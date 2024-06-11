import os
from PIL import Image
from tqdm import tqdm

def create_gif_from_images(directory, output_filename):
    # List all files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    
    # Filter files to ensure they follow the naming convention "0001.png", "0002.png", etc.
    files = [f for f in files if f[:-4].isdigit()]
    
    # Sort the files by their numerical order
    files.sort(key=lambda x: int(x[:-4]))

    # Create a list to hold the images
    images = []

    # Read and append each image
    for file in tqdm(files):
        file_path = os.path.join(directory, file)
        img = Image.open(file_path)
        images.append(img)

    # Check if we have images to create a GIF
    if images:
        # Save as a GIF
        images[0].save(
            directory+output_filename,
            save_all=True,
            append_images=images[1:],
            duration=100,  # Duration between frames in milliseconds
            loop=0        # 0 means loop forever
        )
        print(f"GIF saved as {output_filename}")
    else:
        print("No images found to create a GIF.")

# Usage
directory = '/data1/nuplan/jiale/exp/exp/simulation_FINETUNE/open_loop_boxes/test14-hard/planTF/debug_files/'  # Replace with the path to your directory
# directory = '/data1/nuplan/jiale/exp/exp/simulation/closed_loop_nonreactive_agents/test14-hard/planTF/debug_files/'  # Replace with the path to your directory
output_filename = 'output.gif'  # Replace with the desired output filename
create_gif_from_images(directory, output_filename)
