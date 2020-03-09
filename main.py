in_file = open("./dataset/local/integration.in")

max_len = []
symbol_dict = {"pad": 0}
data = []
i = 1
for line in in_file.readlines():
    new_line = [0] * 512
    max_len.append(len(line.strip().split(",")))
    j = 0
    for c in line.strip().split(","):
        if c.lower() not in symbol_dict:
            symbol_dict[c.lower()] = i
            i += 1
        new_line[j] = symbol_dict[c.lower()]
        j += 1
    # while len(new_line) < 512:
    #     new_line.append(0)
    data.append(new_line)


import pdb

pdb.set_trace()


out_file = open("./dataset/local/integration.out")
i = 0
for line in out_file.readlines():
    for c in line.strip().split(","):
        if c.lower() not in symbol_dict:
            pdb.set_trace()

