# -*- coding: utf-8 -*-


import json
import tqdm
import torch
import argparse
import deepspeed
import numpy as np
import matplotlib.pyplot as plt
from utility import EarlyStopping
from torch.utils.data import DataLoader
from reformer_pytorch import ReformerLM


def cycle(loader):
    while True:
        for data in loader:
            yield data

def add_argument():
    parser=argparse.ArgumentParser(description='sym-math')

    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# constants according to the paper
dim = 512
depth = 6
heads = 8
max_seq_len = 512
leraning_rate = 1e-4
batch_size = 256
optimizer = torch.optim.Adam

# constants not revealed in the paper
emb_dim = 512
batch_num = 10000
grad_accum = 4


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
train_data = dataset[: int(data_len * 0.8)]
valid_data = dataset[int(data_len * 0.8) : int(data_len * 0.9)]
test_data = dataset[int(data_len * 0.9) :]

#train_loader = cycle(DataLoader(train_data, batch_size=int(batch_size/grad_accum), shuffle=True))
#valid_loader = cycle(DataLoader(valid_data, batch_size=batch_size, shuffle=True))

print("Building model ....")

model = ReformerLM(
    num_tokens=len(token_dict),
    max_seq_len=max_seq_len,
    emb_dim=emb_dim,
    depth=depth,
    heads=heads,
    dim=dim,
    fixed_position_emb=True,
    lsh_dropout = 0.1,
    ff_dropout = 0.1,
    post_attn_dropout = 0.1,
    layer_dropout = 0.1,
).to(device)

cmd_args = add_argument()
model, optim, train_loader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(), trainin_data = train_data)
model = torch.nn.DataParallel(model)
#optim = optimizer(model.parameters(), lr=leraning_rate)
loss_ftn = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

print("Start training!")

early_stopping = EarlyStopping(patience=100, verbose=True)

torch.cuda.empty_cache()

train_losses = []
valid_losses = []
for i in tqdm.tqdm(range(batch_num), mininterval=10, desc="training"):

    model.train()
    optim.zero_grad()

    for _ in range(grad_accum):
        batch = next(train_loader)
        source = batch[:, 0, :]
        target = batch[:, 1, :].contiguous()
        output = model(source).contiguous()
        loss = loss_ftn(output.view(-1, len(token_dict)), target.view(-1))
        loss.backward()

    loss_value = loss.item()
    print(f"training loss: {loss_value}")
    train_losses.append(loss_value)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()

    if i % 4 == 0:
        model.eval()
        with torch.no_grad():
            batch = next(valid_loader)
            source = batch[:, 0, :]
            target = batch[:, 1, :].contiguous()
            output = model(source).contiguous()

            loss = loss_ftn(output.view(-1, len(token_dict)), target.view(-1))
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
