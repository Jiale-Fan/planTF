from tutorials.utils.tutorial_utils import construct_nuboard_hydra_paths
import hydra
from nuplan.planning.script.run_nuboard import main as main_nuboard



# Location of paths with all nuBoard configs
nuboard_hydra_path = "./config/nuboard"

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=nuboard_hydra_path)

# Compose the configuration
cfg = hydra.compose(config_name="/planTF_nuboard.yaml", overrides=[
    'scenario_builder=nuplan_challenge',  # set the database (same as simulation) used to fetch data for visualization
    f'simulation_path=null',  # nuboard file path, if left empty the user can open the file inside nuBoard
    f'hydra.searchpath=["pkg://nuplan.planning.script.config.common","pkg://nuplan.planning.script.experiments"]'
])


# Run nuBoard
main_nuboard(cfg)