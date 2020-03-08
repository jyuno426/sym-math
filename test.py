# -*- coding: utf-8 -*-

from sympy import *
from datetime import date
from core import *
import time
import sys

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
if __name__ == "__main__":
    for i in range(1000):
        for a in generate_raw_data("ode1"):
            traverse = []
            for arg in preorder_traversal(a):
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
                # elif "Derivative" in class_name:
                #     print(arg.args)
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
                        print(a)
                        raise Exception()
                        # print(arg.func)

            print(traverse)
