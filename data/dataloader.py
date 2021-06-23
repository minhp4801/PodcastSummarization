import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

def generate_splits(config):
    """Partitions the dataset into the the k-fold train/validation set and the 
    leave-out test set. 
    
    Args:
        config: 
            A dict containing the network and dataset configuration details for this experiment.
    Returns:
        kfold_split:
            A list of the portion of the dataset that has been set aside for training / validation.
        train_splits:
            A list of lists that indicates the indices of kfold_split which are for training, at each iteration.
        val_splits:
            A list of list that indicates the indices of kfold_split which are for validation, at each iteration.
        test_split:
            A list of the leave-out portion of the dataset, only used for final testing after we've considered
            several models.
    """

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
    """Determine the 95th percentile of summary token lengths. This is to mitigate the 
    chances of a very long summary dominating the input to BERT (leaving no room for the
    source text).

    Args:
        config: 
            A dict containing the network and dataset configuration details for this experiment.
        tokenizer:
            A preinitialized BERT tokenizer that we will be using to determine the number of 
            tokens a summary has.
    Returns:
        summary_len:
            An integer that serves as the maximum length for our first sentence in our BERT input.
            For example, if our model's maximum input was 10, and we determine the max summary
            length to be 3, our input would look something like:
            
            [CLS] <3 tokens> [SEP] <5 tokens>
    """
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
    """A custom dataset class that will allow us to set the sentence pair split in BERT.
    
    Set up mainly to work with the transcripts of the Spotify 100,000 Podcast dataset.

    Attributes:
        summary_token_len:
            The maximum length of our summary. We subtract 2 to account for the special 
            tokens [CLS] and [SEP].
        corpus_token_len:
            The maximum length of our corpus. This is determined by subtracting 
        summary_with_scores:
            A pandas dataframe containing both the summaries and their associated scores.
        corpora_dir:
            The directory where all the podcast transcripts are.
        tokenizer:
            The tokenizer we will use for generating our tokens.
    """

    def __init__(self, config, tokenizer, classes, summary_token_len):
        """

        Args:
            config: 
                A dict containing the network and dataset configuration details for this experiment.
            tokenizer:
                A preinitialized tokenizer for generating tokens.
            classes:
                A list of the names associated with the classes. Makes finding the corpus file easier.
            summary_token_len:
                The 95th percentile of summary token lengths. Determines where we separate 
                summaries and corpora. 
        """
        
        # Set our summary / corpus input token split. Subtract 2 from summary_token_len to account
        # for the special characters [CLS] and [SEP].
        self.summary_token_len = summary_token_len - 2
        self.corpus_token_len = config['dataset']['input_max'] - summary_token_len

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
        """Slight modification to the typical __getitem__, we have to manually set the
        separation between summary and corpus.
        """
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