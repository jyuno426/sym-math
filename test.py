# -*- coding: utf-8 -*-


import json
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from utility import EarlyStopping
from torch.utils.data import DataLoader
from reformer_pytorch import ReformerLM
from datetime import date
from core import *
import time
import sys

binary_map = {
    "Add": _add,
    "Mul": _mul,
    "Pow": _pow,
}

unary_map = {
    "exp": _exp,
    "log": _log,
    "sin": _sin,
    "cos": _cos,
    "tan": _tan,
    "asin": _asin,
    "acos": _acos,
    "atan": _atan,
    "sinh": _sinh,
    "cosh": _cosh,
    "tanh": _tanh,
    "asinh": _asinh,
    "acosh": _acosh,
    "atanh": _atanh,
    "sqrt": _sqrt,
}

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
batch_num = 5000  # 4 epochs for 160000 training sets


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=2)
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(len(target)):
            j = 0
            check = True
            while j < len(target[i]):
                if pred[i][j] != target[i][j]:
                    check = False
                    break
                if target[i][j] == 0:
                    break
            if check:
                correct += 1
    return correct / len(target)


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
            lsh_dropout=0.1,
            ff_dropout=0.1,
            post_attn_dropout=0.1,
            layer_dropout=0.1,
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
            lsh_dropout=0.1,
            ff_dropout=0.1,
            post_attn_dropout=0.1,
            layer_dropout=0.1,
        ).to(device)

    def forward(self, source, target):
        return self.decoder(target, keys=self.encoder(source))


def parse_string_to_expr(string):
    root = Node()
    empty_node_stack = [root]

    arg_list = string.split(",")

    i = 0
    n = len(arg_list)

    # print("n: ", n)
    while i < n:
        # print("i: ", i)
        arg = arg_list[i]
        # print(arg)
        # if len(empty_node_stack) <= 0:
        #     print(i, n, arg_list)
        assert len(empty_node_stack) > 0
        node = empty_node_stack.pop()

        if arg == "p" or arg == "m":
            res = 0
            j = i + 1
            while j < n and arg_list[j] in "0123456789":
                res *= 10
                res += int(arg_list[j])
                j += 1
            i = j
            node.data = S(res if arg == "p" else -res)
        elif arg == "x":
            node.data = x
            i += 1
        elif arg == "c":
            node.data = c
            i += 1
        elif arg == "f":
            node.data = f
            i += 1
        elif arg == "g":
            node.data = diff(f, x)
            i += 1
        elif arg == "E":
            node.data = E
            i += 1
        elif arg == "pi":
            node.data = pi
            i += 1
        elif arg in unary_map:
            node.data = unary_map[arg]
            empty_node_stack.append(node.add_child(Node()))
            i += 1
        elif arg in binary_map:
            node.data = binary_map[arg]
            left = node.add_child(Node())
            right = node.add_child(Node())
            empty_node_stack.append(right)
            empty_node_stack.append(left)
            i += 1
        else:
            print(string)
            print(arg)
            raise Exception("parse error: " + arg)

    return root.get_expr().doit()


if __name__ == "__main__":
    model = MyModel().to(device)
    model = torch.nn.DataParallel(model)
    optim = optimizer(model.parameters(), lr=learning_rate)
    loss_ftn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum").to(device)

    model.load_state_dict(torch.load("checkpoint.pt"))

    token_dict = {"pad": 0}
    inverse_token_dict = {0: "pad"}
    token_idx = 1

    print("Loading ....")

    # preprocess input data
    in_lines = []
    with open("./dataset/local/integration.in", "r") as f:
        for i, line in enumerate(f.readlines()):
            token_list = line.strip().split(",")
            if i >= 200000:
                in_lines.append(token_list)
            for i, _token in enumerate(token_list):
                token = _token.lower()
                if token not in token_dict:
                    token_dict[token] = token_idx
                    inverse_token_dict[token_idx] = token
                    token_idx += 1

    for i in len(in_lines):
        token_list = in_lines[i]
        in_lines[i] = [token_dict[token] for token in token_list]

    j = len(in_lines)

    print("total:", j)
    cnt = 0
    for i in range(j):
        # print(i)
        # input_string, output_string = generate_data("ode1")
        # # input_string, output_string = in_lines[i], out_lines[i]
        # _f = parse_string_to_expr(output_string)
        # _diff_f = diff(_f, x)
        # _diff_eq = parse_string_to_expr(input_string)

        # if test_equal(subs_func(subs_func(_diff_eq, f, _f), diff(f, x), _diff_f)):
        #     cnt += 1
        #     print("yeah")
        #     # print(simplify(diff(_f, x) - _diff_f))
        # print(i)
        # input_string, output_string = in_lines[i], out_lines[i]

        source = torch.tensor([in_lines[i]])
        output = model()
        try:
            _f = parse_string_to_expr(output_string)
            _diff_f = parse_string_to_expr(input_string)
            if test_equal(diff(_f, x) - _diff_f):
                cnt += 1
                print("yeah")
            else:
                print(diff(_f, x))
                print(_diff_f)
                # print(simplify(diff(_f, x) - _diff_f))
        except:
            pass

    print(str(round(cnt / j * 100)) + "%")
