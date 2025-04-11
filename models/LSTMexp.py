import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from pymongo import MongoClient

def load_data_from_mongo(source="all", **kwargs):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["EnigmaticCodes"]
    data = []

    if source in ("atbash", "all"):
        data += [(d["Encripted"], d["Original"]) for d in db.Atbash.find(**kwargs)]
    
    if source in ("caesar", "all"):
        data += [(d["Encripted"], d["Original"]) for d in db.Caesar.find(**kwargs)]
    
    if source in ("pl", "all"):
        data += [(d["Encripted"], d["Original"]) for d in db.PL.find(**kwargs)]
    
    return data


# ==== Підготовка алфавіту ====
ALL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

CHARS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(ALL_LETTERS)
char2idx = {ch: i for i, ch in enumerate(CHARS)}
idx2char = {i: ch for ch, i in char2idx.items()}

def encode_seq(seq):
    return [char2idx[SOS_TOKEN]] + [char2idx[c] for c in seq] + [char2idx[EOS_TOKEN]]

def decode_seq(indices):
    return ''.join([idx2char[i] for i in indices if idx2char[i] not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]])

# ==== Датасет ====
class CipherDataset(Dataset):
    def __init__(self, pairs, max_len=20):
        self.data = pairs
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        enc, dec = self.data[idx]
        enc_idx = encode_seq(enc.lower())[:self.max_len]
        dec_idx = encode_seq(dec.lower())[:self.max_len]
        
        enc_idx += [char2idx[PAD_TOKEN]] * (self.max_len - len(enc_idx))
        dec_idx += [char2idx[PAD_TOKEN]] * (self.max_len - len(dec_idx))
        
        return torch.tensor(enc_idx), torch.tensor(dec_idx)

# ==== Модель ====
class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)

        # Якщо є target (режим тренування)
        if tgt is not None:
            embedded_tgt = self.embedding(tgt[:, :-1])
            outputs, _ = self.decoder(embedded_tgt, (hidden, cell))
            logits = self.fc(outputs)
            return logits
        else:
            # Inference mode — генеруємо по 1 символу
            inputs = torch.full((src.size(0), 1), char2idx[SOS_TOKEN], dtype=torch.long, device=src.device)
            outputs = []
            for _ in range(20):  # Max sequence length
                embedded = self.embedding(inputs)
                out, (hidden, cell) = self.decoder(embedded, (hidden, cell))
                logit = self.fc(out[:, -1, :])
                pred = torch.argmax(logit, dim=1, keepdim=True)
                outputs.append(pred)
                inputs = pred
            return torch.cat(outputs, dim=1)

# ==== Приклад тренування ====
def train_model(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    
    loss_fn = nn.CrossEntropyLoss(ignore_index=char2idx[PAD_TOKEN])
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x, y)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# ==== Inference ====
def predict(model, word):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([encode_seq(word.lower()) + [char2idx[PAD_TOKEN]] * 10], device=device)
        output = model(x)
        return decode_seq(output[0].cpu().numpy())



# ==== Використання ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Приклад даних: (зашифроване, розшифроване)
sample_data = load_data_from_mongo()
dataset = CipherDataset(sample_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)




model = Seq2SeqLSTM(vocab_size=len(char2idx), embedding_dim=64, hidden_dim=128).to(device)
train_model(model, loader, epochs=50)

# Тест
print(predict(model, "uryyb"))     # -> "hello"
print(predict(model, "svool"))     # -> "hello"
print(predict(model, "ellohay"))   # -> "hello"
