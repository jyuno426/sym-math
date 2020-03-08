# -*- coding: utf-8 -*-

from sympy import Symbol, symbols, Function

__all__ = ["x", "c", "d", "f", "g"]

x = Symbol("x", real=True, positive=True, nonzero=True)
c, d = symbols("c d", real=True, nonzero=True)  # constants
f = Function("f")(x)
g = Function("g")(x)
