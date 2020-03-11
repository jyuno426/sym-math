# -*- coding: utf-8 -*-

from utility import *
from sympy import *
import numpy as np

from .constants import *
from .tree import *


__all__ = ["generate_data"]

internal_node_size = 15
terminals = set([S(i) if i != 0 else x for i in range(-5, 6)])
binary_operations = set([_add, _sub, _mul, _div])
unary_operations = set(
    [
        _exp,
        _log,
        _sqrt,
        _sin,
        _cos,
        _tan,
        _asin,
        _acos,
        _atan,
        _sinh,
        _cosh,
        _tanh,
        _asinh,
        _acosh,
        _atanh,
    ]
)

c1 = ["Rational", "Half"]
c2 = ["Integer", "Symbol", "One", "NegativeOne", "Zero", "Exp1", "Pi"]
c3 = [
    "asinh",
    "acosh",
    "atanh",
    "asin",
    "acos",
    "atan",
    "acot",
    "asec",
    "acsc",
    "sinh",
    "cosh",
    "tanh",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "Abs",
    "exp",
    "log",
    "sqrt",
]  # one args / order of c3 elements important!!!!! we should compare asinh first then sinh then sin
c4 = ["Pow"]  # two args
c5 = ["Add", "Mul"]  # multiple args

invalid_list = ["oo", "I", "Dummy", "nan", "zoo", "conjugate"]


# dp table for tree_count
tree_count_table = [
    [-1 for j in range(internal_node_size + 2)] for i in range(internal_node_size + 2)
]


def tree_count(empty_node_count, internal_node_count):
    """
    Calculate the number of tree representations in terms of
    combinations of terminals, unary and binary operations defiend in constants.py.
    @params{
        empty_node_count:       # of empty nodes wating to be allocated.
        internal_node_count:    # of internal_nodes not determined yet.
    }
    """

    enc = empty_node_count
    inc = internal_node_count
    l_t = len(terminals)
    l_u = len(unary_operations)
    l_b = len(binary_operations)

    assert enc >= 0 and inc >= 0

    if tree_count_table[enc][inc] == -1:
        # if dp value is not set yet
        if inc == 0:
            tree_count_table[enc][inc] = l_t ** enc

        elif enc == 0:
            tree_count_table[enc][inc] = 0

        else:
            c_t = l_t * tree_count(enc - 1, inc)
            c_u = l_u * tree_count(enc, inc - 1)
            c_b = l_b * tree_count(enc + 1, inc - 1)

            tree_count_table[enc][inc] = c_t + c_u + c_b

    return tree_count_table[enc][inc]


def sample_intenral_node_size():
    """
    Sample internal_node_size uniformly in [1, max_internal_node_size].
    "uniformly" means combinatorial uniformness according to tree_count
    """
    node_size_list = range(1, internal_node_size)
    probs = [tree_count(1, node_size) for node_size in node_size_list]
    probs_sum = sum(probs)
    probs = [p / probs_sum for p in probs]
    return np.random.choice(node_size_list, p=probs)


def generate_random_tree():
    """
    Generate random tree (math-expression) uniformly which consists of
    terminals, unary and binary operations defiend in constants.py.
    Returns Tree object.
    "uniformly" means combinatorial uniformness
    """

    tree = Tree(Node())
    empty_node_list = [tree.root]

    if internal_node_size is None:
        internal_node_count = sample_intenral_node_size()
    else:
        internal_node_count = internal_node_size

    while len(empty_node_list) + internal_node_count > 0:
        assert internal_node_count >= 0

        total_count = tree_count(len(empty_node_list), internal_node_count)

        # sampling
        nodes, probs = [], []

        for v in terminals:
            count = tree_count(len(empty_node_list) - 1, internal_node_count)
            probs.append(count / total_count)
            nodes.append({"data": v, "child_count": 0})

        if internal_node_count > 0:
            for v in unary_operations:
                count = tree_count(len(empty_node_list), internal_node_count - 1)
                probs.append(count / total_count)
                nodes.append({"data": v, "child_count": 1})

            for v in binary_operations:
                count = tree_count(len(empty_node_list) + 1, internal_node_count - 1)
                probs.append(count / total_count)
                nodes.append({"data": v, "child_count": 2})

        sampled_node = np.random.choice(nodes, p=probs)

        selected_empty_node = empty_node_list.pop(
            np.random.choice(range(len(empty_node_list)))
        )

        selected_empty_node.data = sampled_node["data"]
        for _ in range(sampled_node["child_count"]):
            empty_node_list.append(selected_empty_node.add_child(Node()))

        if sampled_node["child_count"] > 0:
            internal_node_count -= 1

    return tree


def generate_data(data_type):
    _in, _out = _generate_data(data_type)
    return [expr_to_preorder(_in), expr_to_preorder(_out)]


def _generate_data(data_type):
    if data_type == "integration":
        while True:
            tree = generate_random_tree()
            ftn = tree.get_expr()
            if test_real(ftn, x):
                ftn = drop_number(reduce_expr(tree.get_expr()), [x])
                # ftn = reduce_expr(tree.get_expr())
                if not test_valid(ftn):
                    continue

                try:
                    deriv = diff(ftn, x)
                except:
                    slack_message("derivative error", data_type)
                    continue

                deriv = reduce_expr(diff(ftn, x))
                if not test_valid(deriv):
                    continue

                return [deriv, ftn]

    elif data_type == "ode1":
        while True:
            tree = generate_random_tree()
            ftn = tree.get_expr()
            if test_real(ftn, x):
                # replace random leaf by c
                np.random.choice(tree.get_leaf_list()).data = c

                ftn = reduce_coefficient(tree.get_expr(init=True), c, x)
                if not test_valid(ftn):
                    continue

                sol = solve_expr_1(_sub(f, ftn), S(0), c)
                if not test_valid(sol):
                    continue

                # reduce shoulde be done after solve_expr_1
                ftn = reduce_expr(ftn)
                if not test_valid(ftn):
                    continue
                try:
                    diff_eq = diff(sol, x)
                except:
                    slack_message("derivative error", data_type)
                    continue

                diff_eq = reduce_expr(subs_func(diff_eq, diff(f, x), g))
                if not test_valid(diff_eq):
                    continue
                try:
                    diff_eq = fraction(diff_eq)[0]
                except:
                    pass

                return [diff_eq, ftn]
        pass

    elif data_type == "ode2":
        pass

    else:
        raise Exception("Invalid data_type: " + str(data_type))


def expr_to_preorder(expr):
    res = []

    for elem in preorder_traverse(expr):
        string = str(elem)
        if check_number(string):
            if string[0] == "-":
                res.append("m")
                string = string[1:]
            elif string[0] == "+":
                res.append("p")
                string = string[1:]
            else:
                res.append("p")
            for c in string:
                res.append(c)
        else:
            res.append(string)

    return ",".join(res)


def preorder_traverse(expr):
    traverse = []
    class_name = expr.func.__name__

    if any(name in class_name for name in c1):
        traverse.append(Mul.__name__)
        left, right = str(expr).split("/")
        traverse.append(left)
        traverse.append(Pow.__name__)
        traverse.append(right)
        traverse.append("-1")

    elif any(name in class_name for name in c2):
        traverse.append(str(expr))

    elif class_name == "f":
        traverse.append("f")

    elif class_name == "g":
        traverse.append("g")

    elif any(name in class_name for name in c3):
        assert len(expr.args) == 1
        for name in c3:
            if name in class_name:
                traverse.append(name)
                traverse += preorder_traverse(expr.args[0])
                break

    elif "Pow" in class_name:  # c4
        assert len(expr.args) == 2
        traverse.append("Pow")
        traverse += preorder_traverse(expr.args[0])
        traverse += preorder_traverse(expr.args[1])

    elif any(name in class_name for name in c5):
        assert len(expr.args) >= 2
        for name in c5:
            if name in class_name:
                traverse += [name] * (len(expr.args) - 1)
                for arg in expr.args:
                    traverse += preorder_traverse(arg)
                break
    else:
        raise Exception("preorder_traverse error")

    return traverse


# def preorder_traverse(expr):
#     traverse = []
#     for arg in preorder_traversal(expr):
#         class_name = arg.func.__name__
#         res = None

#         if any(name in class_name for name in c1):
#             traverse.append(Mul.__name__)
#             left, right = str(arg).split("/")
#             traverse.append(left)
#             traverse.append(Pow.__name__)
#             traverse.append(right)
#             traverse.append("-1")
#         elif any(name in class_name for name in c2):
#             traverse.append(str(arg))
#         elif class_name == "f":
#             traverse.append("f")
#         elif class_name == "g":
#             traverse.append("g")
#         else:
#             check = False
#             for name in c3:
#                 if name in class_name:
#                     traverse.append(name)
#                     check = True
#                     break
#             if not check:
#                 slack_message(
#                     str(arg.func)
#                     + "\n"
#                     + str(arg)
#                     + "\n"
#                     + str(traverse)
#                     + "\n"
#                     + str(expr)
#                 )
#                 print(arg.func)
#                 print(arg)
#                 print(traverse)
#                 print(expr)
#                 raise Exception()

#     res = []
#     for elem in traverse:
#         string = str(elem)
#         if check_number(string):
#             if string[0] == "-":
#                 res.append("m")
#                 string = string[1:]
#             elif string[0] == "+":
#                 res.append("p")
#                 string = string[1:]
#             else:
#                 res.append("p")
#             for c in string:
#                 res.append(c)
#         else:
#             res.append(string)

#     return ",".join(res)

