import matplotlib
import matplotlib.pyplot as plt
import numpy as np

f1 = open("./dataset/local/src-train.txt", "w")
f2 = open("./dataset/local/src-val.txt", "w")
f3 = open("./dataset/local/src-test.txt", "w")
f4 = open("./dataset/local/tgt-train.txt", "w")
f5 = open("./dataset/local/tgt-val.txt", "w")
f6 = open("./dataset/local/tgt-test.txt", "w")

# mappp = {
#     "0": "zero",
#     "1": "one",
#     "2": "two",
#     "3": "three",
#     "4": "four",
#     "5": "five",
#     "6": "six",
#     "7": "seven",
#     "8": "eight",
#     "9": "nine",
#     ",": " ",
# }


# def mapp(string):
#     string = string.strip()
#     for key, val in mappp.items():
#         string = string.replace(key, val)
#     return string


# def unmapp(string):
#     string = string.strip()
#     for key, val in mappp.items():
#         string = string.replace(val, key)
#     return string


with open("./dataset/local/integration.in", "r") as f:
    lines = f.readlines()
    lines = lines[np.random.permutation(len(lines))]
    for i in range(len(lines) * 0.8):
        f1.write(f.readline().replace(",", " "))
    for i in range(len(lines) * 0.1):
        f2.write(f.readline().replace(",", " "))
    for i in range(len(lines) * 0.1):
        f3.write(f.readline().replace(",", " "))

with open("./dataset/local/integration.out", "r") as f:
    lines = f.readlines()
    lines = lines[np.random.permutation(len(lines))]
    for i in range(len(lines) * 0.8):
        f4.write(f.readline().replace(",", " "))
    for i in range(len(lines) * 0.1):
        f5.write(f.readline().replace(",", " "))
    for i in range(len(lines) * 0.1):
        f6.write(f.readline().replace(",", " "))

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
    range(8 * batch_size, 8 * batch_size * len(valid_loss) + 1, 8 * batch_size)
)

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
