import torch
import pandas as pd
from transformers import AutoModel

import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn

class BERT_FineTune(nn.Module):
    def __init__(self, bert):

        super(BERT_FineTune, self).__init__()

        # Load the pretrained bert.
        self.bert = bert

        # These layers are the ones we are adding.
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 4)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x