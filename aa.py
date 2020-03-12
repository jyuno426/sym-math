import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

f1 = open("./dataset/local/src-train.txt", "w")
f2 = open("./dataset/local/src-val.txt", "w")
f3 = open("./dataset/local/src-test.txt", "w")
f4 = open("./dataset/local/tgt-train.txt", "w")
f5 = open("./dataset/local/tgt-val.txt", "w")
f6 = open("./dataset/local/tgt-test.txt", "w")

with open("./dataset/local/integration.in", "r") as f:
    with open("./dataset/local/integration.in", "r") as g:
        in_lines = f.readlines()
        out_lines = g.readlines()
        assert len(in_lines) == len(out_lines) and len(in_lines) == 220000
        random_indexes = np.random.permutation(len(in_lines))

        i = 0
        while i < 210000:
            f1.write(in_lines[random_indexes[i]].strip().replace(",", " ") +
                     '\n')
            f4.write(out_lines[random_indexes[i]].strip().replace(",", " ") +
                     '\n')
            i += 1
        while i < 215000:
            f2.write(in_lines[random_indexes[i]].strip().replace(",", " ") +
                     '\n')
            f5.write(out_lines[random_indexes[i]].strip().replace(",", " ") +
                     '\n')
            i += 1
        while i < 220000:
            f3.write(in_lines[random_indexes[i]].strip().replace(",", " ") +
                     '\n')
            f6.write(out_lines[random_indexes[i]].strip().replace(",", " ") +
                     '\n')
            i += 1

import sys

sys.exit()

with open("train_loss", "r") as f:
    train_loss = np.array([line.strip() for line in f.readlines()])
    train_loss = train_loss.astype(np.float)

with open("valid_loss", "r") as f:
    valid_loss = np.array([line.strip() for line in f.readlines()])
    valid_loss = valid_loss.astype(np.float)

with open("train_acc", "r") as f:
    train_acc = np.array([line.strip() for line in f.readlines()])
    train_acc = train_acc.astype(np.float)

with open("valid_acc", "r") as f:
    valid_acc = np.array([line.strip() for line in f.readlines()])
    valid_acc = valid_acc.astype(np.float)

batch_size = 128
x1 = np.array(range(batch_size, len(train_loss) * batch_size + 1, batch_size))
x2 = np.array(
    range(8 * batch_size, 8 * batch_size * len(valid_loss) + 1,
          8 * batch_size))

plt.title("Integration")
plt.plot(x1, train_loss, label="train_loss")
plt.plot(x2, valid_loss, label="valid_loss")
# plt.plot(x2, train_acc, label="train_acc")
# plt.plot(x2, train_acc, label="valid_acc")

plt.xlabel("# of data")
plt.ylabel("loss")
# plt.ylabel("acc")

plt.legend()
# plt.show()
plt.savefig("res_loss.png")
