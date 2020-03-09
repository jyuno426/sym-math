# -*- coding: utf-8 -*-

import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from utility import EarlyStopping
from torch.utils.data import DataLoader
from reformer_pytorch import ReformerLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constants according to the paper
dim = 512
depth = 6
heads = 8
max_seq_len = 512
leraning_rate = 1e-4
batch_size = 256
optimizer = torch.optim.Adam

# constants not revealed in the paper
emb_dim = 128
epochs = int(1e5)


in_data = []
out_data = []

token_dict = {"pad": 0}
token_idx = 1

print("Preprocessing ....")

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

dataset = [(in_data[i], out_data[i]) for i in np.random.permutation(data_len)]
dataset = torch.tensor(dataset).to(device)

# divide dataset
train_data = dataset[: int(data_len * 0.72)]
valid_data = dataset[int(data_len * 0.72) : int(data_len * 0.9)]
test_data = dataset[int(data_len * 0.9) :]

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

print("Building model ....")

model = ReformerLM(
    num_tokens=len(token_dict),
    max_seq_len=max_seq_len,
    emb_dim=emb_dim,
    depth=depth,
    heads=heads,
    dim=dim,
    fixed_position_emb=True,
)
model = torch.nn.DataParallel(model)
model = model.to(device)
optim = optimizer(model.parameters(), lr=leraning_rate)
loss_ftn = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

print("Start training!")

avg_train_losses = []
avg_valid_losses = []

early_stopping = EarlyStopping(patience=20, verbose=True)

torch.cuda.empty_cache()

for epoch in tqdm.tqdm(range(1, epochs + 1), mininterval=10, desc="training"):
    train_losses = []
    valid_losses = []

    model.train()

    for batch in train_loader:
        source = batch[:, 0, :]
        target = batch[:, 1, :]
        output = model(source)
        loss = loss_ftn(output, target)
        loss.backward()
        optim.step()
        optim.zero_grad()
        loss_value = loss.item()
        # print(f"training loss: {loss_value}")
        train_losses.append(loss_value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            source = batch[:, 0, :]
            target = batch[:, 1, :]
            output = model(source)
            loss = loss_ftn(output, target)
            loss_value = loss.item()
            # print(f"valid loss: {loss_value}")
            valid_losses.append(loss_value)

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(epochs))
    print_msg = (
        f"[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] "
        + f"train_loss: {train_loss:.5f} "
        + f"valid_loss: {valid_loss:.5f}"
    )
    print(print_msg)

    # early_stopping needs the validation loss to check if it has decresed,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

    torch.cuda.empty_cache()

# load the last checkpoint with the best model
# model.load_state_dict(torch.load("checkpoint.pt"))
