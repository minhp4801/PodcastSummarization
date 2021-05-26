
import yaml
from PodcastSummarization.data.dataloader import *


def run(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
if __name__ == "__main__":
    with open("../experiments/barebones.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    