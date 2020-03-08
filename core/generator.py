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
    "Abs",
    "Add",
    "Mul",
    "Pow",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
]

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


def generate_data(data_type="integration"):
    if data_type == "integration":
        while True:
            tree = generate_random_tree()
            ftn = tree.get_expr()
            if test_real(ftn, x):
                ftn = reduce_expr(drop_number(tree.get_expr().doit(), [x]))
                deriv = reduce_expr(diff(ftn, x))
                assert test_valid(ftn) and test_valid(deriv)
                return [expr_to_preorder(deriv), expr_to_preorder(ftn)]
    elif data_type == "ode1":
        while True:
            tree = generate_random_tree()
            ftn = tree.get_expr()
            if test_real(ftn, x):
                # replace random leaf by c
                np.random.choice(tree.get_leaf_list()).data = c

                ftn = reduce_coefficient(tree.get_expr(init=True), c, x)
                assert test_valid(ftn)
                try:
                    solution = solve_expr(f - ftn, c)
                except TimeoutError:
                    # slack_message("solve timeout", data_type)
                    continue
                except Exception as e:
                    slack_message("solve error: " + str(e), data_type)
                    continue
                if not test_valid(solution):
                    continue

                ftn = reduce_expr(ftn)
                if not test_valid(ftn):
                    continue

                diff_eq = reduce_expr(
                    subs_derivative(fraction(diff(solution.doit(), x))[0], g)
                )
                if not test_valid(diff_eq):
                    continue

                return [expr_to_preorder(diff_eq), expr_to_preorder(ftn)]
        pass
    elif data_type == "ode2":
        pass
    else:
        raise Exception("Invalid data_type: " + str(data_type))


def expr_to_preorder(expr):
    traverse = []
    for arg in preorder_traversal(expr):
        class_name = str(arg.func)
        res = None

        if any(name in class_name for name in c1):
            traverse.append("Mul")
            left, right = str(arg).split("/")
            traverse.append(left)
            traverse.append("Pow")
            traverse.append(right)
            traverse.append("-1")
        elif any(name in class_name for name in c2):
            traverse.append(str(arg))
        elif class_name == "f":
            traverse.append("f")
        elif class_name == "g":
            traverse.append("g")
        else:
            check = False
            for name in c3:
                if name in class_name:
                    traverse.append(name)
                    check = True
                    break
            if not check:
                print(arg.func)
                print(arg)
                print(traverse)
                print(expr)
                raise Exception()

    return ",".join(traverse)

