# -*- coding: utf-8 -*-

from sympy import *
from datetime import date
from utility import *
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
    # in_file = open("./dataset/2020-03-08/integration.in", "r")
    # out_file = open("./dataset/2020-03-08/integration.out", "r")

    # in_lines = [line.strip() for line in in_file.readlines()]
    # out_lines = [line.strip() for line in out_file.readlines()]
    # i = len(in_lines)
    # j = len(out_lines)
    # assert i == j
    # in_file.close()
    # out_file.close()

    j = 100

    print("total:", j)
    cnt = 0
    for i in range(j):
        print(i)
        input_string, output_string = generate_data("ode1")
        # input_string, output_string = in_lines[i], out_lines[i]
        _f = parse_string_to_expr(output_string)
        _diff_f = diff(_f, x)
        _diff_eq = parse_string_to_expr(input_string)

        if test_equal(subs_func(subs_func(_diff_eq, f, _f), diff(f, x), _diff_f)):
            cnt += 1
            print("yeah")
            # print(simplify(diff(_f, x) - _diff_f))
        # print(i)
        # input_string, output_string = generate_data("integration")
        # # input_string, output_string = in_lines[i], out_lines[i]
        # _f = parse_string_to_expr(output_string)
        # _diff_f = parse_string_to_expr(input_string)
        # if test_equal(diff(_f, x) - _diff_f):
        #     cnt += 1
        #     print("yeah")
        # else:
        #     print(diff(_f, x))
        #     print(_diff_f)
        #     # print(simplify(diff(_f, x) - _diff_f))

    print(str(round(cnt / j * 100)) + "%")
