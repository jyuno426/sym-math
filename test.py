# -*- coding: utf-8 -*-

import json
import tqdm
# import torch
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from utility import *
# from torch.utils.data import DataLoader
# from reformer_pytorch import ReformerLM
from datetime import date
from core import *
import time
import sys
import traceback

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


def parse_string_to_expr(string):
    root = Node()
    empty_node_stack = [root]

    arg_list = string.split(" ")

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
            assert 1 == 0
            # print(string)
            # print(arg)
            raise Exception("parse error: " + arg)

    # print(string)
    # for node in empty_node_stack:
    #     print(node.data)
    assert len(empty_node_stack) == 0

    try:
        res = root.get_expr().doit()
    except:
        print("holy")

    return res


def mywrite(string, end=None):
    with open("dataset/log-test.txt", "a+") as f:
        if end is None:
            f.write(string + "\n")
        else:
            f.write(string + end)


def mywrite_mapping(string, end=None):
    with open("dataset/log-test-mapping.txt", "a+") as f:
        if end is None:
            f.write(string + "\n")
        else:
            f.write(string + end)


if __name__ == "__main__":
    _file = "test"
    src = []
    tgt = []
    pred = []
    pred_mapping = []
    beam_size = 10
    with open("dataset/local/src-" + _file + ".txt", "r") as f:
        for line in f.readlines():
            _line = line.strip()
            src.append(_line)
    with open("dataset/local/tgt-" + _file + ".txt", "r") as f:
        for line in f.readlines():
            _line = line.strip()
            tgt.append(_line)
    with open("dataset/local/pred-" + _file + ".txt", "r") as f:
        for line in f.readlines():
            _line = line.strip()
            if len(pred) == 0 or len(pred[-1]) == beam_size:
                pred.append([])
            pred[-1].append(_line)
    # with open("dataset/local/pred-" + _file + "-mapping.txt", "r") as f:
    #     for line in f.readlines():
    #         _line = line.strip()
    #         if len(pred_mapping) == 0 or len(pred_mapping[-1]) == beam_size:
    #             pred_mapping.append([])
    #         pred_mapping[-1].append(_line)

    assert len(src) == len(tgt) and len(src) == len(
        pred)  # and len(src) == len(pred_mapping)

    n = len(src)
    mywrite("total: " + str(n))
    # mywrite_mapping("total: " + str(n))
    data_error_cnt = 0

    exact_cnt = 0
    success_cnt = 0
    incorrect_cnt = 0
    parse_error_cnt = 0
    evaluation_error_cnt = 0

    # exact_cnt_mapping = 0
    # success_cnt_mapping = 0
    # incorrect_cnt_mapping = 0
    # parse_error_cnt_mapping = 0
    # evaluation_error_cnt_mapping = 0

    incorrects = []
    incorrects_mapping = []
    test_cnt = 30

    for i in range(n):
        assert len(pred[i]) == beam_size
        # assert len(pred_mapping[i]) == beam_size

        try:
            _diff_f = parse_string_to_expr(src[i])
        except Exception as e:
            # mywrite("src_error:", e)
            data_error_cnt += 1
            mywrite(
                str(data_error_cnt) + "-" + str(exact_cnt) + "-" +
                str(success_cnt) + "-" + str(incorrect_cnt) + "-" +
                str(parse_error_cnt) + "-" + str(evaluation_error_cnt) + "/" +
                str(i + 1))
            # mywrite_mapping(
            #     str(data_error_cnt) + "-" + str(exact_cnt_mapping) + "-" +
            #     str(success_cnt_mapping) + "-" + str(incorrect_cnt_mapping) +
            #     "-" + str(parse_error_cnt_mapping) + "-" +
            #     str(evaluation_error_cnt_mapping) + "/" + str(i + 1))
            continue

        data_domain_error_cnt = 0
        numerics = []
        tgt_vals = []
        while data_domain_error_cnt < 200 and len(numerics) < test_cnt:
            numeric = np.random.uniform(-100, 100)
            try:
                tgt_val = get_numeric(_diff_f, x, numeric)
                tgt_vals.append(tgt_val)
                numerics.append(numeric)
                data_domain_error_cnt = 0
            except:
                data_domain_error_cnt += 1

        if len(numerics) != test_cnt:
            data_error_cnt += 1
            mywrite(
                str(data_error_cnt) + "-" + str(exact_cnt) + "-" +
                str(success_cnt) + "-" + str(incorrect_cnt) + "-" +
                str(parse_error_cnt) + "-" + str(evaluation_error_cnt) + "/" +
                str(i + 1))
            # mywrite_mapping(
            #     str(data_error_cnt) + "-" + str(exact_cnt_mapping) + "-" +
            #     str(success_cnt_mapping) + "-" + str(incorrect_cnt_mapping) +
            #     "-" + str(parse_error_cnt_mapping) + "-" +
            #     str(evaluation_error_cnt_mapping) + "/" + str(i + 1))
            continue

        # ----------------------------------------------------------------

        cur_errors = []
        exact = False
        success = False
        parse_success = False
        evaluation_success = False
        for j in range(beam_size):
            if pred[i][j] == tgt[i]:
                exact = True
                break

            try:
                _f = parse_string_to_expr(pred[i][j])
            except:
                # parse error
                continue
            parse_success = True

            try:
                __diff__f = diff(_f, x)
            except:
                # evaluation error
                continue

            errors = []
            for ii in range(test_cnt):
                try:
                    pred_val = get_numeric(__diff__f, x, numerics[ii])
                except:
                    # evaluation error
                    break

                dd = np.absolute(tgt_vals[ii] - pred_val)
                errors.append(dd)
                # ss = np.absolute(tgt_vals[ii]) + np.absolute(pred_val)
                # if ss < 1e-10:  # if sum is 0
                #     errors.append(dd)
                # else:
                #     errors.append(dd / ss)

            if len(errors) == test_cnt:
                evaluation_success = True

                check = True
                for error in errors:
                    if error >= 1e-4:
                        check = False
                        break
                if check:
                    success = True
                    break
                else:
                    cur_errors.append(errors)

        if exact:
            exact_cnt += 1
        elif success:
            success_cnt += 1
        elif not parse_success:
            parse_error_cnt += 1
        elif not evaluation_success:
            evaluation_error_cnt += 1
        else:
            incorrect_cnt += 1
            incorrects.append(cur_errors)

        mywrite(
            str(data_error_cnt) + "-" + str(exact_cnt) + "-" +
            str(success_cnt) + "-" + str(incorrect_cnt) + "-" +
            str(parse_error_cnt) + "-" + str(evaluation_error_cnt) + "/" +
            str(i + 1))

        # ----------------------------------------------------------------

        # cur_errors = []
        # exact = False
        # success = False
        # parse_success = False
        # evaluation_success = False
        # for j in range(beam_size):
        #     if pred_mapping[i][j] == tgt[i]:
        #         exact = True
        #         break

        #     try:
        #         _f = parse_string_to_expr(pred_mapping[i][j])
        #     except:
        #         # parse error
        #         # print("holy")
        #         # print(pred_mapping[i][j])
        #         continue
        #     parse_success = True

        #     try:
        #         __diff__f = diff(_f, x)
        #     except:
        #         # evaluation error
        #         continue

        #     errors = []
        #     for ii in range(test_cnt):
        #         try:
        #             pred_val = get_numeric(_diff_f, x, numerics[ii])
        #         except:
        #             # evaluation error
        #             break

        #         dd = np.absolute(tgt_vals[ii] - pred_val)
        #         errors.append(dd)
        #         # ss = np.absolute(tgt_vals[ii]) + np.absolute(pred_val)
        #         # if ss < 1e-10:  # if sum is 0
        #         #     errors.append(dd)
        #         # else:
        #         #     errors.append(dd / ss)

        #     if len(errors) == test_cnt:
        #         evaluation_success = True

        #         check = True
        #         for error in errors:
        #             if error >= 1e-4:
        #                 check = False
        #                 break
        #         if check:
        #             success = True
        #             break
        #         else:
        #             cur_errors.append(errors)

        # if exact:
        #     exact_cnt_mapping += 1
        # elif success:
        #     success_cnt_mapping += 1
        # elif not parse_success:
        #     parse_error_cnt_mapping += 1
        # elif not evaluation_success:
        #     evaluation_error_cnt_mapping += 1
        # else:
        #     incorrect_cnt_mapping += 1
        #     incorrects_mapping.append(cur_errors)

        # mywrite_mapping(
        #     str(data_error_cnt) + "-" + str(exact_cnt_mapping) + "-" +
        #     str(success_cnt_mapping) + "-" + str(incorrect_cnt_mapping) + "-" +
        #     str(parse_error_cnt_mapping) + "-" +
        #     str(evaluation_error_cnt_mapping) + "/" + str(i + 1))

    for errors in incorrects:
        mywrite(str(np.average(errors)))
    # for errors in incorrects_mapping:
    #     mywrite_mapping(str(np.average(errors)))