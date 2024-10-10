import yaml
import torch

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_device():
    """Set the device to use (CPU or GPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"
