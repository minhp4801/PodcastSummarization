import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AdamW
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, random_split
from dataloader import SpotifyPodcastDataset
from model_arch import BERT_FineTune
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

if __name__ == "__main__":
    device = torch.device('cuda')
    epochs = 5

    # Load the dataset.
    dataset = SpotifyPodcastDataset(csv_file="./data/RatedSummaryComprehensive.csv", root_dir="./data/podcast_transcripts/", tokenizer="bert-large-uncased")

    # Load bert-large-uncased as the base model for fine-tuning.
    bert = AutoModel.from_pretrained('bert-large-uncased')

    # Freeze the layers of BERT, we're only concerned with the last layers we provide.
    for param in bert.parameters():
        param.requires_grad = False

    model = BERT_FineTune(bert)
    model = model.to(device)

    # Load the AdamW optimizer.
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Load Cross Entropy Loss.
    cross_entropy = nn.CrossEntropyLoss()

    # Split the dataset into a 70/15/15 split. 
    # TODO: Change this to k-fold cross-validation!
    split = random_split(dataset, [3956, 698, 698])

    train_split = split[0]
    val_split = split[1]
    test_split = split[2]

    train_sampler = RandomSampler(train_split)
    train_dataloader = DataLoader(train_split, sampler=train_sampler, batch_size=8)

    val_sampler = SequentialSampler(val_split)
    val_dataloader = DataLoader(val_split, sampler=val_sampler, batch_size=8)

    test_sampler = SequentialSampler(test_split)
    test_dataloader = DataLoader(test_split, sampler=test_sampler, batch_size=8)

    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        train_loss, _ = train(train_dataloader)
        valid_loss, _ = evaluate(val_dataloader)

        # Store the best model for evaluation.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    # Make predictions on the test data.
    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path))
    
    test_loss, _ = evaluate(test_dataloader)

def train(train_dataloader):
    # Run our model in training mode.
    model.train()
    
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):

        batch = [r.to(device) for r in batch]

        sent_id = batch['input_ids']
        mask = batch['attention_mask']
        labels = batch['score']

        # Clear gradients from the previous iterations.
        model.zero_grad()

        preds = model(sent_id, mask)

        # After calculating the loss, update the optimizer.
        loss = cross_entropy(preds, labels)

        total_loss = total_loss + loss.item()

        loss.backward()

        # Clips the gradient at 1.0 to avoid exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds = preds.detach().cpu().numpy()

        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def evaluate(val_dataloader):
    # Run our model in evaluation mode.
    model.eval()

    total_loss = 0

    total_preds = []
    
    for step, batch in enumerate(val_dataloader):
        
        batch = [t.to(device) for t in batch]

        # Capture the relevant inputs from the dataloader.
        sent_id = batch['input_ids']
        mask = batch['attention_mask']
        labels = batch['score']

        # Evaluate the predictions of the model without calculating gradients.
        with torch.no_grad():
            preds = model(sent_id, mask)

            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)
    
    avg_loss = total_loss / len(val_dataloader)

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds