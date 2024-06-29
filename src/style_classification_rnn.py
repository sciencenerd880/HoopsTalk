import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from nltk import word_tokenize


style_map = {
    0: "david",
    1: "trump"
}

MAX_ENCODED_LEN = 50
device = torch.device('mps')

with open("./models/vocab2index_style_classification.json", 'r') as f:
    vocab2index = json.load(f)

vocab_size = len(vocab2index)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(RNNModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, 128)
        
        self.bilstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True, dropout=0.1, num_layers=2)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(64, output_size)


    def forward(self, x):
        output = self.embedding(x)
        output, _ = self.bilstm(output)

        output = output[:, -1, :]  # Get the output of the last time step
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)
    
        return output
    

rnn_model = RNNModel(vocab_size=vocab_size, output_size=1).to(device)
rnn_model.load_state_dict(torch.load("./models/style_classification_rnn.pth", map_location=device))
rnn_model.eval()

# predict
def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', "\"\'"))
    text = re.sub(r'\s+', ' ', text)
    text = word_tokenize(text.lower().strip())
    text = [token.strip() for token in text if token.strip() != ""]

    return text


def encode_sentence(text, vocab2index, max_len=50):
    encoded = np.zeros(max_len, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(max_len, len(enc1)) # if above max len, cut the rest
    encoded[:length] = enc1[:length]

    return encoded


def predict(model, text):
    text = clean_text(text)
    encoded_text = torch.tensor(encode_sentence(text, vocab2index, MAX_ENCODED_LEN)).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(encoded_text)
        predicted_proba = F.sigmoid(outputs)
        predicted = torch.round(predicted_proba)
        predicted_style = style_map[int(predicted.data[0][0].cpu().numpy())]

    return predicted_proba, predicted, predicted_style


df = pd.read_csv("./data/text/GPT4o/personified_15words_with_tst.csv")

trump_text = df["trump_caption"].apply(clean_text).apply(lambda x: encode_sentence(x, vocab2index, max_len=MAX_ENCODED_LEN))
tst_text = df["tst_trump_text"].apply(clean_text).apply(lambda x: encode_sentence(x, vocab2index, max_len=MAX_ENCODED_LEN))

trump_text = torch.tensor(np.vstack(trump_text)).to(device)
tst_text = torch.tensor(np.vstack(tst_text)).to(device)

labels = [1] * len(df)

with torch.no_grad():
    trump_outputs = rnn_model(trump_text).squeeze()
    print(F.sigmoid(trump_outputs))
    trump_pred = torch.round(F.sigmoid(trump_outputs)).cpu().numpy().astype(int)
    tst_outputs = rnn_model(tst_text).squeeze()
    print(F.sigmoid(tst_outputs))
    tst_pred = torch.round(F.sigmoid(tst_outputs)).cpu().numpy().astype(int)

    trump_accuracy = accuracy_score(labels, trump_pred)
    tst_accuracy = accuracy_score(labels, tst_pred)

print(trump_accuracy)
print(tst_accuracy)
