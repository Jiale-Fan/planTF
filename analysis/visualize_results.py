import torch
from visualization import plot_sample_elements

from src.models.planTF.lightning_trainer import LightningTrainer
import pickle
import hydra
import os
from src.features.nuplan_feature import NuplanFeature

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# use hydra to initialize the pytorch model 

nuboard_hydra_path = "../config/model"
# Initialize configuration management system
hydra.initialize(config_path=nuboard_hydra_path)

# Compose the configuration
cfg = hydra.compose(config_name="/planTF.yaml", overrides=[
    # f'hydra.searchpath=["pkg://nuplan.planning.script.config.common","pkg://nuplan.planning.script.experiments"]'
])

torch_model = hydra.utils.instantiate(cfg)
torch_model.to(torch.device("cuda:0"))

# Load the model from a checkpoint
checkpoint_path = '/data1/nuplan/jiale/exp/exp/training/planTF/plantf_1kmodescon_100k/checkpoints/last.ckpt'
model = LightningTrainer.load_from_checkpoint(checkpoint_path, model = torch_model)

with open('./debug_files/nuplan_feature_data.pkl', 'rb') as f:
        data = pickle.load(f)

out = model(data)
deserialized_input = NuplanFeature.deserialize(data=data)[0]

plot_sample_elements(deserialized_input, out[0])
