import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AdamW
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from dataloader import SpotifyPodcastDataset
from model_arch import BERT_FineTune
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

if __name__ == "__main__":
    device = torch.device('cuda')
    epochs = 5
    dataset = SpotifyPodcastDataset(csv_file="./data/RatedSummaryComprehensive.csv", root_dir="./data/podcast_transcripts/", tokenizer="bert-large-uncased")

    bert = AutoModel.from_pretrained('bert-large-uncased')

    for param in bert.parameters():
        param.requires_grad = False

    model = BERT_FineTune(bert)

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    cross_entropy = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        train_loss, _ = train()

        valid_loss, _ = evaluate()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

def train(train_dataloader):
    model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):

        batch = [r.to(device) for r in batch]

        sent_id = batch['input_ids']
        mask = batch['attention_mask']
        labels = batch['score']

        model.zero_grad()

        preds = model(sent_id, mask)

        loss = cross_entropy(preds, labels)

        total_loss = total_loss + loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds = preds.detach().cpu().numpy()

        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def evaluate(val_dataloader):
    
    model.eval()

    total_loss = 0

    total_preds = []
    
    for step, batch in enumerate(val_dataloader):
        
        batch = [t.to(device) for t in batch]

        sent_id = batch['input_ids']
        mask = batch['attention_mask']
        labels = batch['score']

        with torch.no_grad():
            preds = model(sent_id, mask)

            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)
    
    avg_loss = total_loss / len(val_dataloader)

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

