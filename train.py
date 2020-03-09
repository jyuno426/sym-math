# -*- coding: utf-8 -*-

import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from reformer_pytorch import ReformerLM

GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data


class Seq2Seq(torch.nn.Module):
    def __init__(
        self,
        num_tokens,
        max_seq_len,
        emb_dim=128,
        dim=512,
        depth=6,
        heads=8,
        fixed_position_emb=True,
    ):
        super(Seq2Seq, self).__init__()

        self.reformer = ReformerLM(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            emb_dim=emb_dim,
            dim=dim,
            depth=depth,
            heads=heads,
            fixed_position_emb=True,
        )

    def forward(self, source):
        return torch.nn.Softmax(self.reformer(source), dim=1)


# constants according to the paper
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
dataset = torch.tensor(dataset).long().to(device)

# divide dataset
train_data = dataset[: int(data_len * 0.72)]
validation_data = dataset[int(data_len * 0.72) : int(data_len * 0.9)]
test_data = dataset[int(data_len * 0.9) :]

train_loader = cycle(DataLoader(train_data, batch_size=batch_size, shuffle=True))
validation_loader = cycle(
    DataLoader(validation_data, batch_size=batch_size, shuffle=True)
)

print("Building model ....")

model = Seq2Seq(num_tokens=len(token_dict), max_seq_len=max_seq_len).to(device)
optim = optimizer(model.parameters(), lr=leraning_rate)
loss_ftn = torch.nn.CrossEntropyLoss(ignore_index=0)

print("Start training!")

for i in tqdm.tqdm(range(epochs), mininterval=10, desc="training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_loader)
        source = batch[:, 0, :]
        target = batch[:, 1, :]
        output = model(source)
        loss = loss_ftn(output.view(-1), target.view(-1))
        loss.backward()

    print(f"training loss: {loss.item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            source, target = next(validation_loader)
            output = model(source)
            loss = loss_ftn(output.view(-1), target.view(-1))
            print(f"validation loss: {loss.item()}")

