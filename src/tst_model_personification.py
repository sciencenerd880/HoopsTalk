import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import word_tokenize


device = torch.device('mps')

VOCAB2INDEX_PATH = "./models/vocab2index.json"
TST_MODEL_PATH = "./models/rnngru_style_transfer.pth"
PERSONIFIED_CSV = "./data/text/GPT4o/personified_15words.csv"


def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', "\"\'"))
    text = re.sub(r'\s+', ' ', text)
    text = word_tokenize(text.lower().strip())
    text = [token.strip() for token in text if token.strip() != ""]

    return text


with open(VOCAB2INDEX_PATH, 'r') as f:
    vocab2index = json.load(f)

index2vocab = {v: k for k, v in vocab2index.items()}

def encode_sentence(text, vocab2index, max_len=50):
    encoded = np.zeros(max_len, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(max_len, len(enc1)) # if above max len, cut the rest
    encoded[:length] = enc1[:length]

    return encoded

class StyleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, style_size, pretrained_embeddings=None):
        super(StyleEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Optionally, freeze embeddings
        self.style_embed = nn.Embedding(1, style_size)  # Only one style
        self.rnn = nn.GRU(embed_size + style_size, hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, style):
        x = self.embedding(x).to(device)
        style_embedding = self.style_embed(style).unsqueeze(1).expand_as(x)
        x = torch.cat([x, style_embedding], dim=2)
        output, _ = self.rnn(x)
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.decoder(output)
        return output

vocab_size = len(vocab2index)
embed_size = 100
hidden_size = 256
style_size = 100

tst_model = StyleEmbeddingModel(vocab_size, embed_size, hidden_size, style_size).to(device)
tst_model.load_state_dict(torch.load(TST_MODEL_PATH, map_location=device))
tst_model.eval()

personified = pd.read_csv(PERSONIFIED_CSV)


text_to_convert = personified["gen_commentary"].apply(clean_text)
text_to_convert = text_to_convert.apply(lambda x: encode_sentence(x, vocab2index, max_len=50))

encoded_text = np.vstack(text_to_convert)
encoded_tensor = torch.tensor(encoded_text, dtype=torch.long).to(device)

def predict(model, encoded_tensor):
    encoded_tensor = encoded_tensor.unsqueeze(0)
    style_label = torch.tensor([0], dtype=torch.long).repeat(encoded_tensor.shape[0]).to(device)
    with torch.no_grad():
        outputs = model(encoded_tensor, style_label)
        predicted_ids = torch.argmax(outputs, dim=-1).squeeze().tolist()
        styled_text = " ".join([index2vocab.get(idx, 'UNK') for idx in predicted_ids])
    
    return styled_text

predicted_text = []
for enc_tensor in encoded_tensor:
    predicted_text.append(predict(tst_model, enc_tensor))


personified["tst_trump_text"] = predicted_text

personified.to_csv("./data/text/GPT4o/personified_15words_with_tst.csv")
