# -*- coding: utf-8 -*-

import json
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from utility import EarlyStopping
from torch.utils.data import DataLoader
from reformer_pytorch import ReformerLM


def cycle(loader):
    while True:
        for data in loader:
            yield data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# constants according to the paper
dim = 512
depth = 6
heads = 8
max_seq_len = 512
learning_rate = 2.5e-5
batch_size = 128
optimizer = torch.optim.Adam

# constants not revealed in the paper
emb_dim = 512
batch_num = 5000 # 4 epochs for 160000 training sets


in_data = []
out_data = []

token_dict = {"pad": 0}
token_idx = 1

print("Loading ....")

# preprocess input data
with open("./dataset/local/integration.in", "r") as f:
    for line in f.readlines():
        _data = [0] * max_seq_len
        token_list = line.strip().split(",")
        for i, _token in enumerate(token_list):
            token = _token.lower()
            if token not in token_dict:
                token_dict[token] = token_idx
                token_idx += 1
            _data[i] = token_dict[token]
        in_data.append(_data)

# preprocess output data
with open("./dataset/local/integration.out", "r") as f:
    for line in f.readlines():
        _data = [0] * max_seq_len
        token_list = line.strip().split(",")
        for i, _token in enumerate(token_list):
            token = _token.lower()
            if token not in token_dict:
                token_dict[token] = token_idx
                token_idx += 1
            _data[i] = token_dict[token]
        out_data.append(_data)

assert len(in_data) == len(out_data)
data_len = len(in_data)

print("Total tokens: ", len(token_dict))
print("Total dataset: ", data_len)

print("Preprocessing ....")

dataset = [(in_data[i], out_data[i]) for i in np.random.permutation(data_len)]
dataset = torch.tensor(dataset)

# divide dataset
train_data = dataset[: int(data_len * 0.8)]
valid_data = dataset[int(data_len * 0.8) : int(data_len * 0.9)]
test_data = dataset[int(data_len * 0.9) :]

train_loader = cycle(DataLoader(train_data, batch_size=int(batch_size), shuffle=True))
valid_loader = cycle(DataLoader(valid_data, batch_size=batch_size, shuffle=True))

print("Building model ....")

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = ReformerLM(
            num_tokens=len(token_dict),
            max_seq_len=max_seq_len,
            emb_dim=emb_dim,
            depth=depth,
            heads=heads,
            dim=dim,
            fixed_position_emb=True,
            return_embeddings=True,
            lsh_dropout = 0.1,
            ff_dropout = 0.1,
            post_attn_dropout = 0.1,
            layer_dropout = 0.1,
        ).to(device)

        self.decoder = ReformerLM(
            num_tokens=len(token_dict),
            max_seq_len=max_seq_len,
            emb_dim=emb_dim,
            depth=depth,
            heads=heads,
            dim=dim,
            fixed_position_emb=True,
            causal=True,
            lsh_dropout = 0.1,
            ff_dropout = 0.1,
            post_attn_dropout = 0.1,
            layer_dropout = 0.1,
        ).to(device)

    def forward(self, source, target):
        return self.decoder(target, keys=self.encoder(source))


model = MyModel().to(device)
model = torch.nn.DataParallel(model)
optim = optimizer(model.parameters(), lr=learning_rate)
loss_ftn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum").to(device)

# model.load_state_dict(torch.load('checkpoint.pt'))

print("Start training!")

early_stopping = EarlyStopping(patience=100, verbose=True)

torch.cuda.empty_cache()

train_losses = []
valid_losses = []
for i in tqdm.tqdm(range(batch_num), mininterval=10, desc="training"):

    model.train()

    for _ in range(1):
        batch = next(train_loader).to(device)
        output = model(batch[:, 0, :], batch[:, 1, :])
        loss = loss_ftn(output.contiguous().view(-1, len(token_dict)), batch[:, 1, :].contiguous().view(-1))
        loss = loss.mean()
        loss.backward()

    loss_value = loss.item()
    print(f"training loss: {loss_value}")
    train_losses.append(loss_value)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optim.step()
    optim.zero_grad()

    if i % 4 == 0:
        model.eval()
        with torch.no_grad():
            batch = next(valid_loader).to(device)
            output = model(batch[:, 0, :], batch[:, 1, :])
            loss = loss_ftn(output.contiguous().view(-1, len(token_dict)), batch[:, 1, :].contiguous().view(-1))
            loss = loss.mean()

            loss_value = loss.item()
            print(f"valid loss: {loss_value}")
            valid_losses.append(loss_value)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(loss_value, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    torch.cuda.empty_cache()

json.dump(train_losses, open("train_losses.json", "w"))
json.dump(valid_losses, open("valid_losses.json", "w"))
# load the last checkpoint with the best model
# model.load_state_dict(torch.load("checkpoint.pt"))
