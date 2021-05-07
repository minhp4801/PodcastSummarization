import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.model_selection import train_test_split, KFold
from data import SpotifyPodcastDataset

if __name__ == "__main__":
    with open("./config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Get our list of episode enumerations.
    with open(config['data']['episodes'], 'r') as f:
        episodes = [episode.strip('\n') for episode in f.readlines()]

    kfold_split, evaluation_split = train_test_split(episodes, train_size=config['data']['train_classes'], shuffle=True)

    kf = KFold(n_splits=config['data']['k-folds'], shuffle=False)
    for train_ind, test_ind in kf.split(kfold_split):
        # train_ind is a list of indices
        # test_ind is also a list of indices

        train = [kfold_split[i] for i in train_ind]
        test = [kfold_split[i] for i in test_ind]

        train_set = SpotifyPodcastDataset(config, train)
        test_set = SpotifyPodcastDataset(config, test)
    
    eval_set = SpotifyPodcastDataset(config, evaluation_split)


        

    