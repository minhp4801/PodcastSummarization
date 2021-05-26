import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

def generate_splits(config):
    with open(config['dataset']['podcast_names'], 'r') as f:
        episodes = [episode.strip('\n') for episode in f.readlines()]
    
    kfold_split, test_split = train_test_split(episodes, train_size=config['dataset']['train_classes'], shuffle=True)

    train_splits = []
    val_splits = []

    kf = KFold(n_splits=config['dataset']['k-folds'], shuffle=False)

    for train_ind, val_ind in kf.split(kfold_split):
        train_splits.append(train_ind)
        val_splits.append(val_ind)
    
    return kfold_split, train_splits, val_splits, test_split

def get_summary_length(config, tokenizer):

    df = pd.read_csv(config['dataset']['summary_with_scores'])

    lengths = []

    # Loop through summaries and store the tokenized lengths in array,
    for _, row in df.iterrows():
        text = row['Summary']
        tokens = tokenizer(text)
        lengths.append(len(tokens['input_ids']))
    
    # Use np.percentile to get the 95th percentile of lengths (avoids outliers).
    summary_len = np.percentile(lengths, 95)

    # Return 95th percentile length.
    return int(summary_len)

class SpotifyPodcastDataset(Dataset):
    def __init__(self, config, tokenizer, classes, summary_token_len):
        '''
        config (dict): A parsed YAML file containing program configuration.
        tokenizer (modelf): The pretrained tokenizer that we will be using to tokenize the strings.
        classes (list): A list of classes that will be included as part of this dataset.
        summary_token_len (int): The chosen length that the summaries will be truncated / padded to.
        '''
        
        # Set our summary / corpus input token split.
        self.summary_token_len = summary_token_len
        self.corpus_token_len = config['dataset']['input_max']

        # Load the CSV with the specified summaries and their scores.
        df = pd.read_csv(config['dataset']['summary_with_scores'])
        self.summary_with_scores = df[df.Number.isin(classes)].reset_index(drop=True)

        # Set up our directory of podcast transcripts.
        self.corpora_dir = config['dataset']['corpora_dir']

        # Initialize our pretrained tokenizer.
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.summary_with_scores)
    
    def __getitem__(self, idx):
        '''
        idx (int, list, or tensor): The specific index / indices to return.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the summary and the score that corresponds to idx from the csv.
        summary = self.summary_with_scores.iloc[idx, 1]
        score = self.summary_with_scores.iloc[idx, 2]

        # Get our summary tokens, limiting the max length to what is speficied in the config file.
        # We include special tokens [CLS] and [SEP] here for compliance with BERT's input.
        summary_tokens = self.tokenizer(summary, max_length=self.summary_token_len, padding='max_length', truncation=True)
        
        # Extract the associated corpus as a string. 
        corpus_name = os.path.join(
            self.corpora_dir, 
            self.summary_with_scores.iloc[idx, 0] + '.txt'
        )
        with open(corpus_name, 'r', encoding='latin-1') as f:
            corpus = f.read()

        # Get our corpus tokens. We do not need the special tokens here since we will append these 
        # tokens to the summary tokens.
        corpus_tokens = self.tokenizer(corpus, max_length=self.corpus_token_len, padding='max_length', truncation=True, add_special_tokens=False)

        # Combine the summary and corpus tokens. The resulting tokens should be in the format:
        # '[CLS] <Summary> [SEP] <Corpus>' 
        seq = torch.tensor(summary_tokens['input_ids'] + corpus_tokens['input_ids'])
        mask = torch.tensor(summary_tokens['attention_mask'] + corpus_tokens['attention_mask'])

        # Create our sample dictionary that has everything our network needs for training / testing.
        sample = {'input_ids': seq, 'attention_mask': mask, 'score': score}

        return sample