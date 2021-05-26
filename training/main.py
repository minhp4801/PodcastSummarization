
import time
import datetime
import numpy as np
from transformers import BertForSequenceClassification, AdamW, get_linear_scheduler_with_warmup

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(config, device, train_dataloader, val_dataloader):

    # Initialize a new model for each time we call this function.
    model = BertForSequenceClassification.from_pretrained(
        config['network']['finetune-base'],
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_steps = len(train_dataloader) * config['network']['epochs']

    get_linear_scheduler_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(config['network']['epochs']):
        t0 = time.time()

        total_train_loss = 0

        model.train()
        # TRAINING 
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['score'].to(device)

            # Zero out the gradients before the next forward pass.
            model.zero_grad()

            # BertForSequenceClassification uses Cross Entropy Loss when provided with multiple classes.
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            total_train_loss += loss.item()

            loss.backward()

            # Clip the gradient to 1.0 to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
        # Calculate the loss over all the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)
    
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        
        # VALIDATION
        for batch in val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['score'].to(device)

            # For the forward pass, don't calculate any gradients.
            with torch.no_grad():
                (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Accumulate the overall accuracy across the batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(val_dataloader)
        validation_time = format_time(time.time() - t0)

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (hh:mm:ss)".format(format_time(time.time()-total_t0)))


def test(config, device, test_dataloader):
    model.eval()

    for batch in test_dataloader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['score'].to(device)

        with torch.no_grad():
            _, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(labels_ids)

if __name__ == "__main__":
    with open("./config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Get our list of episode enumerations.
    with open(config['data']['episodes'], 'r') as f:
        episodes = [episode.strip('\n') for episode in f.readlines()]

    kfold_split, test_split = train_test_split(episodes, train_size=config['data']['train_classes'], shuffle=True)

    # The KFold class will allow us to obtain the indices of our splits for k-fold cross validation.
    kf = KFold(n_splits=config['data']['k-folds'], shuffle=False)

    for train_ind, val_ind in kf.split(kfold_split):
        # kf will give us the indices, which we will use to choose our episodes.
        train = [kfold_split[i] for i in train_ind]
        val = [kfold_split[i] for i in val_ind]

        train_set = SpotifyPodcastDataset(config, train)
        val_set = SpotifyPodcastDataset(config, val)

        train_sampler = RandomSampler(train_set)
        train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=config['network']['batch_size'])

        val_sampler = SequentialSampler(val_set)
        val_dataloader = DataLoader(val_set, sampler=val_sampler, batch_size=config['network']['batch_size'])
        
        train(config, device, train_dataloader, val_dataloader)
    
    test(config, device, test_dataloader)
