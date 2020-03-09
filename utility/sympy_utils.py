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
    "_pow",
    "test_equal",
    "test_real",
    "test_valid",
    "drop_number",
    # "subs_derivative",
    "subs_func",
    "reduce_coefficient",
    "reduce_expr",
    "solve_expr_1",
]

inverse_mapping = {
    exp: log,
    log: exp,
    sin: asin,
    cos: acos,
    tan: atan,
    asin: sin,
    acos: cos,
    atan: tan,
    sinh: asinh,
    cosh: acosh,
    tanh: atanh,
    asinh: sinh,
    acosh: cosh,
    atanh: tanh,
}


def test_equal(expr):
    # with time_limit(0.5):
    #     return simplify(expr) == 0
    try:
        ftn = lambdify([x, c], expr.doit(), "numpy")
    except:
        try:
            if (
                np.absolute(
                    expr.evalf(subs={x: 0.000123453141592, c: 0.000124543211134})
                )
                > 1e-4
            ):
                print("fuck!!!")
                return False
            else:
                return True
        except:
            try:
                with time_limit(10):
                    return simplify(expr.doit()) == 0
            except:
                print("fuck")
                return False

    numeric1 = 0.000123453141592
    numeric2 = 0.000124543211134
    while numeric1 < 1 and numeric2 < 1:
        try:
            if np.abolute(ftn([numeric1, numeric2])) > 1e-4:
                print(np.abolute(ftn([numeric1, numeric2])))
                return False
        except:
            print("fuck11")
            return False
        numeric1 = numeric1 * 10
        numeric2 = numeric2 * 10

    return True


def test_valid(expr):
    try:
        return not any(
            s in str(expr) for s in ["oo", "I", "Dummy", "nan", "zoo", "conjugate"]
        )
    except:
        # slack_message("valid error")
        # when we call str(expr), sympy calculate expr so that it can occur errors
        # such as zero division error.
        return False


def test_real(expr, var):
    try:
        ftn = lambdify(var, expr, "numpy")
    except:
        return False

    numeric = 0.123453141592
    while numeric < 1e4:
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


# def subs_derivative(expr, var):
#     if "Derivative" in str(expr.func):
#         assert "-" not in str(expr.func)
#         return var
#     elif len(expr.args) == 0:
#         return expr
#     else:
#         return expr.func(
#             *[subs_derivative(arg, var) for arg in expr.args], evaluate=False
#         )


def subs_func(expr, func, sub):
    if expr.func == func.func:
        assert "-" not in str(expr.func)
        return sub
    elif len(expr.args) == 0:
        return expr
    else:
        return expr.func(
            *[subs_func(arg, func, sub) for arg in expr.args], evaluate=False
        )


def solve_expr_1(expr, res, var):
    assert var in expr.free_symbols
    assert var not in res.free_symbols

    if expr == var:
        assert len(expr.args) == 0
        return res
    elif expr.func in inverse_mapping:
        assert len(expr.args) == 1
        return solve_expr_1(
            expr.args[0], inverse_mapping[expr.func](res, evaluate=False), var
        )
    elif expr.func == sqrt:
        assert len(expr.args) == 1
        return solve_expr_1(expr.args[0], _pow(res, 2), var)
    elif expr.func == Pow:
        assert len(expr.args) == 2
        assert "numbers" in str(expr.args[1].func)
        return solve_expr_1(expr.args[0], _pow(res, _div(1, expr.args[1])), var)
    elif expr.func == Add:
        assert len(expr.args) == 2
        if var in expr.args[0].free_symbols:
            return solve_expr_1(expr.args[0], _sub(res, expr.args[1]), var)
        else:
            return solve_expr_1(expr.args[1], _sub(res, expr.args[0]), var)
    elif expr.func == Mul:
        assert len(expr.args) == 2
        if var in expr.args[0].free_symbols:
            return solve_expr_1(expr.args[0], _div(res, expr.args[1]), var)
        else:
            return solve_expr_1(expr.args[1], _div(res, expr.args[0]), var)
    else:
        slack_message(
            str(expr.func) + "\n" + str(expr) + "\n" + str(res) + "\n" + str(var)
        )
        raise Exception(
            "solve expr 1 error\n"
            + str(expr.func)
            + "\n"
            + str(expr)
            + "\n"
            + str(res)
            + "\n"
            + str(var)
        )


def reduce_coefficient(expr, coeff, var):
    if coeff not in expr.free_symbols:
        return expr
    elif var not in expr.free_symbols:
        return coeff
    elif len(expr.args) == 0:
        return expr
    else:
        return expr.func(
            *[reduce_coefficient(arg, coeff, var) for arg in expr.args], evaluate=False
        )


def reduce_expr(expr, timeout=3, prob=0.3334):
    # if np.random.rand() < prob:
    #     try:
    #         with time_limit(timeout):
    #             return simplify(expr)
    #     except TimeoutError:
    #         pass
    #     except Exception as e:
    #         slack_message(str(e))
    #         pass
    return expr.doit()


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


def _square(arg):
    return _pow(arg, 2)


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
