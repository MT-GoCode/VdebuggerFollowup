import os
from omegaconf import OmegaConf

config = os.getenv('CONFIG', None)
if config is None:
    print("Must pass a valid config file name in /configs to the CONFIG variable.")

config = OmegaConf.load(f'configs/{config}.yaml')
