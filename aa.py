import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

# f1 = open("./dataset/local/src-train.txt", "w")
# f2 = open("./dataset/local/src-val.txt", "w")
# f3 = open("./dataset/local/src-test.txt", "w")
# f4 = open("./dataset/local/tgt-train.txt", "w")
# f5 = open("./dataset/local/tgt-val.txt", "w")
# f6 = open("./dataset/local/tgt-test.txt", "w")

# with open("./dataset/local/integration.in", "r") as f:
#     with open("./dataset/local/integration.out", "r") as g:
#         in_lines = f.readlines()
#         out_lines = g.readlines()
#         assert len(in_lines) == len(out_lines) and len(in_lines) == 220000
#         random_indexes = np.random.permutation(len(in_lines))

#         i = 0
#         while i < 210000:
#             f1.write(in_lines[random_indexes[i]].strip().replace(",", " ") +
#                      '\n')
#             f4.write(out_lines[random_indexes[i]].strip().replace(",", " ") +
#                      '\n')
#             i += 1
#         while i < 215000:
#             f2.write(in_lines[random_indexes[i]].strip().replace(",", " ") +
#                      '\n')
#             f5.write(out_lines[random_indexes[i]].strip().replace(",", " ") +
#                      '\n')
#             i += 1
#         while i < 220000:
#             f3.write(in_lines[random_indexes[i]].strip().replace(",", " ") +
#                      '\n')
#             f6.write(out_lines[random_indexes[i]].strip().replace(",", " ") +
#                      '\n')
#             i += 1

# import sys

# sys.exit()

# with open("train_loss", "r") as f:
#     train_loss = np.array([line.strip() for line in f.readlines()])
#     train_loss = train_loss.astype(np.float)

# with open("valid_loss", "r") as f:
#     valid_loss = np.array([line.strip() for line in f.readlines()])
#     valid_loss = valid_loss.astype(np.float)

# with open("train_acc", "r") as f:
#     train_acc = np.array([line.strip() for line in f.readlines()])
#     train_acc = train_acc.astype(np.float)

# with open("valid_acc", "r") as f:
#     valid_acc = np.array([line.strip() for line in f.readlines()])
#     valid_acc = valid_acc.astype(np.float)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

with open("log", "r") as f:
    for line in f.readlines():
        if "Step" in line:
            for attr in line.strip().split(";"):
                if "ppl" in attr:
                    train_loss.append(float(attr.split()[-1]))
                elif "acc" in attr:
                    train_acc.append(float(attr.split()[-1]))
        elif "Validation perplexity" in line:
            valid_loss.append(float(line.strip().split()[-1]))
        elif "Validation accuracy" in line:
            valid_acc.append(float(line.strip().split()[-1]))

print(len(valid_acc))
print(len(train_acc))
assert len(valid_acc) == len(valid_loss)
assert len(train_acc) == len(train_loss)
# assert len(train_acc) == 210 * len(valid_acc)
train_acc = train_acc[:len(valid_acc) * 210]
train_loss = train_loss[:len(valid_acc) * 210]

batch_size = 256
x = np.array(range(1, len(valid_acc) + 1))

_train_loss = []
_train_acc = []

for i in range(len(valid_acc)):
    _train_loss.append(np.average(train_loss[210 * i:210 * (i + 1)]))

for i in range(len(valid_acc)):
    _train_acc.append(np.average(train_acc[210 * i:210 * (i + 1)]))

plt.title("Integration")
# plt.plot(x, _train_loss, label="train_loss")
# plt.plot(x, valid_loss, label="valid_loss")
plt.plot(x, _train_acc, label="train_acc")
plt.plot(x, valid_acc, label="valid_acc")

plt.xlabel("epochs")
# plt.ylabel("loss(perplexity)")
plt.ylabel("acc(per each character)")

plt.legend()
# plt.show()
# plt.savefig("res_loss.png")
plt.savefig("res_acc.png")
