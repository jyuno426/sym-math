# -*- coding: utf-8 -*-

from sympy import *
import numpy as np

from .other_utils import *

__all__ = [
    "_add",
    "_sub",
    "_mul",
    "_div",
    "_exp",
    "_log",
    "_sqrt",
    "_sin",
    "_cos",
    "_tan",
    "_asin",
    "_acos",
    "_atan",
    "_sinh",
    "_cosh",
    "_tanh",
    "_asinh",
    "_acosh",
    "_atanh",
    "test_real",
    "test_valid",
    "drop_number",
    "subs_derivative",
    "reduce_coefficient",
    "reduce_expr",
    "solve_expr",
]


def test_valid(expr):
    return not any(
        s in str(expr) for s in ["oo", "I", "Dummy", "nan", "zoo", "conjugate"]
    )


def test_real(expr, var):
    try:
        ftn = lambdify(var, expr, "numpy")
    except:
        return False

    numeric = 0.123453141592
    while numeric < 1e10:
        try:
            value = ftn(numeric)
            if np.isfinite(value) and np.isreal(value):
                return True
        except:
            pass
        numeric = numeric * 10

    return False


def drop_number(expr, vars):
    return expr.as_independent(*vars, as_Add=True)[1]


def subs_derivative(expr, var):
    if "Derivative" in str(expr.func):
        return var
    elif len(expr.args) == 0:
        return expr
    else:
        return expr.func(*[subs_derivative(arg, var) for arg in expr.args])


def reduce_coefficient(expr, coeff, var):
    if coeff not in expr.free_symbols:
        return expr
    elif var not in expr.free_symbols:
        return coeff
    elif len(expr.args) == 0:
        return expr
    else:
        return expr.func(*[reduce_coefficient(arg, coeff, var) for arg in expr.args])


def reduce_expr(expr, timeout=3):
    try:
        with time_limit(timeout):
            return simplify(expr)
    except TimeoutError:
        return expr


def solve_expr(expr, var, timeout=3):
    with time_limit(timeout):
        return solve(expr, var)[0]


def _add(arg1, arg2):
    return Add(arg1, arg2, evaluate=False)


def _sub(arg1, arg2):
    return _add(arg1, _mul(-1, arg2))


def _mul(arg1, arg2):
    return Mul(arg1, arg2, evaluate=False)


def _div(arg1, arg2):
    return _mul(arg1, _pow(arg2, -1))


def _pow(arg1, arg2):
    return Pow(arg1, arg2, evaluate=False)


def _exp(arg):
    return exp(arg, evaluate=False)


def _log(arg):
    return log(arg, evaluate=False)


def _sqrt(arg):
    return sqrt(arg, evaluate=False)


def _sin(arg):
    return sin(arg, evaluate=False)


def _cos(arg):
    return cos(arg, evaluate=False)


def _tan(arg):
    return tan(arg, evaluate=False)


def _asin(arg):
    return asin(arg, evaluate=False)


def _acos(arg):
    return acos(arg, evaluate=False)


def _atan(arg):
    return atan(arg, evaluate=False)


def _sinh(arg):
    return sinh(arg, evaluate=False)


def _cosh(arg):
    return cosh(arg, evaluate=False)


def _tanh(arg):
    return tanh(arg, evaluate=False)


def _asinh(arg):
    return asinh(arg, evaluate=False)


def _acosh(arg):
    return acosh(arg, evaluate=False)


def _atanh(arg):
    return atanh(arg, evaluate=False)
