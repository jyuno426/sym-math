# -*- coding: utf-8 -*-

from datetime import date
from utility import *
from core import *
import traceback
import time
import sys
import os


def build(data_type, dataset_path, n=int(2e4)):
    start_time = time.time()
    slack_message("Start " + data_type + " dataset generation: n=" + str(n), data_type)

    in_path = dataset_path + "/" + data_type + ".in"
    out_path = dataset_path + "/" + data_type + ".out"

    duplicate_check = set()

    try:
        in_file = open(in_path, "r")
        out_file = open(out_path, "r")
        in_lines = [line.strip() for line in in_file.readlines()]
        out_lines = [line.strip() for line in out_file.readlines()]
        i = len(in_lines)
        j = len(out_lines)
        if i == j:
            for k in range(i):
                duplicate_check.add(in_lines[k] + "-" + out_lines[k])
        in_file.close()
        out_file.close()
    except:
        i = 0
        j = 0

    assert i == j

    slack_message("current_cnt: " + str(i), data_type)

    if n < i:
        n = i

    while i < n:
        try:
            with time_limit(10):
                input_string, output_string = generate_data(data_type)
        except RecursionError:
            slack_message("recursion error", data_type)
            continue
        except:
            # handle unknown errors
            trace = str(traceback.format_exc())
            print(trace)
            slack_message("unknown error occurs:\n" + trace, data_type)
            continue

        input_string = input_string.replace("f,x", "f").replace("g,x", "g")
        output_string = output_string.replace("f,x", "f").replace("g,x", "g")

        if not test_valid(input_string + "-" + output_string):
            continue
        if len(input_string.split(",")) > 512:
            continue
        elif len(output_string.split(",")) > 512:
            continue
        elif input_string + "-" + output_string in duplicate_check:
            continue
        else:
            duplicate_check.add(input_string + "-" + output_string)

        in_file = open(in_path, "a+")
        out_file = open(out_path, "a+")
        in_file.write(input_string + "\n")
        out_file.write(output_string + "\n")
        in_file.close()
        out_file.close()

        i += 1

        message = print_progress_bar(
            iteration=i,
            total=n,
            prefix=data_type + " data generation-" + str(i) + "/" + str(n) + ":",
            start_time=start_time,
            current_time=time.time(),
            length=20,
        )

        if i % 1000 == 0:
            slack_message(message, data_type)

    slack_message(data_type + " finished", data_type)


if __name__ == "__main__":
    # today = date.today().strftime("%Y-%m-%d")
    today = "local"
    dataset_path = "./dataset/" + today
    data_type = "integration"
    data_cnt = int(2e4)

    if not os.path.exists("./dataset/"):
        os.makedirs("./dataset/")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    l = len(sys.argv)

    assert 1 <= l <= 3

    try:
        data_type = sys.argv[1]
    except:
        pass
    assert data_type in ["integration", "ode1", "ode2"]

    try:
        data_cnt = int(sys.argv[2])
    except:
        pass
    assert data_cnt >= 0

    build(n=data_cnt, data_type=data_type, dataset_path=dataset_path)
