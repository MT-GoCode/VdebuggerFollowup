import os
from omegaconf import OmegaConf

config_path = os.getenv('CONFIG_PATH', None)
if config_path is None:
    print("Must pass a valid relative yaml filepath to the CONFIG variable.")

config = OmegaConf.load(config_path)

config._metadata_path = config_path