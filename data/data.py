import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings('ignore')

class SpotifyPodcastDataset(Dataset):
    def __init__(self, config, classes, max_token_len):
        '''
        config (dict): A parsed YAML file containing program configuration.
        classes (list): A list of classes that will be included as part of this dataset.
        '''
        # Set our summary / corpus input token split.
        self.summary_token_len = config['data']['summary_tokens']
        self.corpus_token_len = config['data']['input_max'] - self.summary_token_len

        # Load the CSV with the specified summaries and their scores.
        df = pd.read_csv(config['data']['summary_with_scores'])
        self.summary_with_scores = df[df.Number.isin(classes)].reset_index(drop=True)

        # Set up our directory of podcast transcripts.
        self.corpora_dir = config['data']['corpora_dir']

        
        # Initialize our pretrained tokenizer. 
        self.tokenizer = AutoTokenizer.from_pretrained(config['network']['tokenizer'])
        
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
