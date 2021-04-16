import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer 

import warnings
warnings.filterwarnings('ignore')

class SpotifyPodcastDataset(Dataset):
    """Spotify Podcast dataset."""

    def __init__(self, csv_file, root_dir, tokenizer):
        """
        Args:
            csv_file (string): Path to the csv file with the summaries.
            root_dir (string): Directory with all the corpora.
            tokenizer (string): The BERT tokenizer model we will use.
        """
        self.summary_with_scores = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.summary_with_scores)

    def __getitem__(self, idx):
        """
        Args:
            idx (int, list, or tensor): The indices of the dataset to choose.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the summary and the score that corresponds to idx from the csv.
        summary = self.summary_with_scores.iloc[idx, 1]
        score = self.summary_with_scores.iloc[idx, 2]

        # Extract the corpus from the text file that corresponds to the summary.
        corpus_name = os.path.join(
            self.root_dir, 
            self.summary_with_scores.iloc[idx, 0] + '.txt'
        )

        corpus_file = open(corpus_name, 'r', encoding='latin-1')
        corpus = corpus_file.read()
        corpus_file.close()

        # Create our input to the tokenizer using [SEP].
        text = summary + '[SEP]' + corpus

        # Get our tokenized string, truncating at 1024 to comply with BERT-large's input size.
        tokenized_string = self.tokenizer(text, max_length=1024, padding='max_length', truncation=True)
        seq = torch.tensor(tokenized_string['input_ids'])
        mask = torch.tensor(tokenized_string['attention_mask'])

        # Create our data sample with a dictionary that consists of everything that will be passed to the network.
        sample = {'input_ids': seq, 'attention_mask': mask, 'score': score}

        return sample