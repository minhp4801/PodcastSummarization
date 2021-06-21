import time
import datetime
import numpy as np
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from data.dataloader import *
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

def main(yaml_filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(yaml_filepath)

    tokenizer = AutoTokenizer.from_pretrained(config['network']['tokenizer'])

    summary_len = get_summary_length(config, tokenizer)

    kfold_split, train_splits, valid_splits, test_split = generate_splits(config)

    f1_scores = []
    for i in range(len(train_splits)):
        train = [kfold_split[j] for j in train_splits[i]]
        valid = [kfold_split[j] for j in valid_splits[i]]

        train_set = SpotifyPodcastDataset(config, tokenizer, train, summary_len)
        valid_set = SpotifyPodcastDataset(config, tokenizer, valid, summary_len)

        train_sampler = RandomSampler(train_set)
        train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=config['network']['batch_size'])

        valid_sampler = SequentialSampler(valid_set)
        valid_dataloader = DataLoader(valid_set, sampler=valid_sampler, batch_size=config['network']['batch_size'])

        valid_f1 = train_model(config, device, train_dataloader, valid_dataloader)
        f1_scores.append(valid_f1)

    print("")
    print(f"  Average F1 Score for configuration {config['network']['name']}: {np.mean(f1_scores)}")
    
def load_config(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    config : dict
    """
    # Read YAML experiment definition file.
    with open(yaml_filepath, 'r') as stream:
        config = yaml.load(stream)

    return config

def get_parser():
    

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file",
        metavar="FILE",
        required=True,
    )

    return parser

def train_model(config, device, train_dataloader, val_dataloader):
    model = BertForSequenceClassification.from_pretrained(
        config['network']['finetune_base'],
        num_labels=config['dataset']['labels'],
        output_attentions=False,
        output_hidden_states=False
    )

    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_steps = len(train_dataloader) * config['network']['epochs']

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)
    print("")
    print("Running Validation...")
    for epoch in range(config['network']['epochs']):
        print("")
        print(f"Starting epoch {epoch}...")
        
        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['score'].to(device)

            model.zero_grad()

            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = output[0]
            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("  Saving checkpoint of model...")
        torch.save(model.state_dict(), f"./checkpoints/{config['network']['name']}_epoch_{epoch}.model")
        
        print("")
        print("Running Validation...")
        
        t0 = time.time()
        model.eval()


        total_eval_loss = 0
        predictions = []
        true_vals = []

        for batch in val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['score'].to(device)

            with torch.no_grad():
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            loss = output[0]
            logits = output[1]
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        val_f1 = f1_score_func(predictions, true_vals)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        validation_time = format_time(time.time() - t0)
        print("  F1 Score (Weighted): {0:.2f}".format(val_f1))
        print("  Average Valid. Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

    print("")
    print("Training complete!")

    return val_f1

def test(config, device, test_dataloader, model_checkpoint):

    print("")
    print("Beginning testing...")

    model = BertForSequenceClassification.from_pretrained(
        config['network']['finetune_base'],
        num_labels=config['dataset']['labels'],
        output_attentions=False,
        output_hidden_states=False
    )

    model.to(device)

    model.load_state_dict(torch.load(model_checkpoint, map_location=torch.device('cpu')))
    
    # After loading model, start evaluation.
    total_eval_loss = 0
    predictions = []
    true_vals = []

    for batch in test_dataloader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['score'].to(device)

        with torch.no_grad():
            (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    test_f1 = f1_score_func(predictions, true_vals)

    print("")
    print("  F1 Score (Weighted): {0:.2f}".format(test_f1))

# Function to calculate the accuracy of our predictions vs labels
def f1_score_func(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='weighted')

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.filename)
