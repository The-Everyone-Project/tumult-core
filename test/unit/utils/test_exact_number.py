"""Unit tests for :mod:`tmlt.core.utils.exact_number`."""

import itertools
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from fractions import Fraction
from functools import partial
from typing import Any, Callable, Iterator, Tuple
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp
from parameterized import parameterized

from tmlt.core.exceptions import UnsupportedSympyExprError
from tmlt.core.utils import exact_number
from tmlt.core.utils.exact_number import (
    ExactNumber,
    ExactNumberInput,
    _cached_to_sympy,
    _verify_expr_is_an_exact_number,
    _verify_expr_recursively,
)

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present


class TestExactNumber(TestCase):
    """TestCase for ExactNumber."""

    @parameterized.expand(
        [(3, 3), (-3, 3), (0, 0), (-sp.oo, sp.oo), (sp.oo, sp.oo), ("-31", 31)]
    )
    def test_abs(self, value: ExactNumberInput, expected: ExactNumberInput):
        """__abs__ returns the expected value."""
        self.assertEqual(abs(ExactNumber(value)), ExactNumber(expected))

    @parameterized.expand(
        [
            (3, 1, 3),
            (7, 2, Fraction(7, 2)),
            (2, 7, Fraction(2, 7)),
            (0, 3, 0),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, 0),
            (-Fraction(3, 5), -Fraction(5, 10), "6/5"),
        ]
    )
    def test_division(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__truediv__ and __rtruediv__ return the expected values."""
        self.assertEqual(ExactNumber(value1) / value2, expected)
        self.assertEqual(value1 / ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 4),
            (7, 2, 9),
            (2, 7, 9),
            (0, 3, 3),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, -sp.oo),
            (-Fraction(3, 5), -Fraction(5, 10), "-11/10"),
        ]
    )
    def test_addition(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__add__ and __radd__ return the expected values."""
        self.assertEqual(ExactNumber(value1) + value2, expected)
        self.assertEqual(value1 + ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 3),
            (7, 2, 14),
            (2, 7, 14),
            (0, 3, 0),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, -sp.oo),
            (-Fraction(3, 5), -Fraction(5, 10), "3/10"),
        ]
    )
    def test_multiplication(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__mul__ and __rmul__ return the expected values."""
        self.assertEqual(ExactNumber(value1) * value2, expected)
        self.assertEqual(value1 * ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 2),
            (7, 2, 5),
            (2, 7, -5),
            (0, 3, -3),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, sp.oo),
            (-Fraction(3, 5), -Fraction(5, 10), "-1/10"),
        ]
    )
    def test_subtraction(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__sub__ and __rsub__ return the expected values."""
        self.assertEqual(ExactNumber(value1) - value2, expected)
        self.assertEqual(value1 - ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 3),
            (7, 2, 49),
            (2, 7, 128),
            (0, 3, 0),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, 0),
            (sp.Rational(3, 5), Fraction(1, 2), "sqrt(3/5)"),
        ]
    )
    def test_exponentiation(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__pow__ and __rpow__ return the expected values."""
        self.assertEqual(ExactNumber(value1) ** value2, expected)
        self.assertEqual(value1 ** ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3,),
            (-3,),
            (0,),
            (-sp.oo,),
            (sp.oo,),
            ("-31",),
            ("1234/433",),
            (sp.Rational(10000, 3),),
            (Fraction(123456789, 987654321),),
        ]
    )
    def test_to_float(self, value: ExactNumberInput):
        """to_float returns the expected value."""
        exact_value = ExactNumber(value)
        ceil_value = exact_value.to_float(round_up=True)
        floor_value = exact_value.to_float(round_up=False)
        self.assertTrue(
            float(exact_value.expr) + 0.000001
            >= ceil_value
            >= float(exact_value.expr)
            >= floor_value
            >= float(exact_value.expr) - 0.000001
        )

    @parameterized.expand(
        [
            (3,),
            (-3,),
            (0,),
            (-float("inf"),),
            (float("inf"),),
            (-31,),
            (1234 / 433,),
            (10000 / 3,),
            (123456789 / 987654321,),
        ]
    )
    def test_from_float(self, value: float):
        """from_float returns the expected value."""
        ceil_value = ExactNumber.from_float(value, round_up=True)
        floor_value = ExactNumber.from_float(value, round_up=False)
        self.assertTrue(
            value + 0.000001
            >= float(ceil_value.expr)
            >= value
            >= float(floor_value.expr)
            >= value - 0.000001
        )

    @parameterized.expand(
        itertools.combinations([3, 7, 2, 0, sp.oo, 100, -Fraction(3, 5)], 2)
    )
    def test_comparison(self, value1: ExactNumberInput, value2: ExactNumberInput):
        """from_float returns the expected value."""
        comparisons = [
            lambda a, b: a == b,
            lambda a, b: a < b,
            lambda a, b: a > b,
            lambda a, b: a <= b,
            lambda a, b: a >= b,
        ]
        for compare in comparisons:
            expected = bool(compare(ExactNumber(value1).expr, ExactNumber(value2)))
            self.assertEqual(compare(ExactNumber(value1), value2), expected)
            self.assertEqual(compare(value1, ExactNumber(value2)), expected)


# Inputs covering every branch of ExactNumber conversion, valid and invalid.
_CONVERSION_INPUTS = [
    0,
    1,
    5,
    -1,
    10**30,
    -(10**30),
    True,
    False,
    "0",
    "1",
    "-3",
    "0.5",
    "1/3",
    "2 + 7**2",
    "sqrt(5/3)",
    "oo",
    "x + 1",
    "pi + I",
    "1 > 0",
    "not a number!",
    Fraction(1, 2),
    Fraction(-3, 4),
    Fraction(4, 2),
    sp.Integer(0),
    sp.Integer(-3),
    sp.Rational(1, 3),
    sp.S.Half,
    sp.oo,
    -sp.oo,
    sp.pi,
    sp.sqrt(2),
    sp.Integer(2) + sp.sqrt(3),
    sp.Float(3.14),
    sp.I,
    sp.nan,
    sp.zoo,
    sp.symbols("x"),
    float("inf"),
    -float("inf"),
    0.0,
    1.0,
    -1.0,
    3.5,
    float("nan"),
    np.float64(2.0),
    np.int64(3),
    None,
    [1],
    ExactNumber(3),
    ExactNumber("1/3"),
    ExactNumber(sp.oo),
]


def _outcome(func: Callable[[], Any]) -> Tuple[Any, ...]:
    """Returns ("ok", result) or ("error", exception type, message) for func()."""
    try:
        return ("ok", func())
    except Exception as e:
        return ("error", type(e), str(e))


@contextmanager
def _exact_number_slow_path() -> Iterator[None]:
    """Disables the ExactNumber fast paths, so every input takes the full path."""
    with (
        patch.object(exact_number, "_CACHEABLE_INPUT_TYPES", frozenset()),
        patch.object(exact_number, "_SIMPLIFIED_EXACT_SYMPY_TYPES", ()),
    ):
        yield


def _expr_outcome(value: Any) -> Tuple[Any, ...]:
    """Returns the outcome of converting ``value``, including the type of the expr."""
    result = _outcome(lambda: ExactNumber(value).expr)
    if result[0] == "ok":
        return ("ok", result[1], type(result[1]))
    return result


@pytest.mark.parametrize("value", _CONVERSION_INPUTS, ids=repr)
def test_fast_path_matches_slow_path(value: Any):
    """ExactNumber conversion gives the same result or error with the fast paths."""
    with _exact_number_slow_path():
        expected = _expr_outcome(value)
    _cached_to_sympy.cache_clear()
    cold = _expr_outcome(value)
    warm = _expr_outcome(value)
    assert cold == expected
    assert warm == expected
    if expected[0] == "ok":
        # The fast path must still produce a simplified expression.
        assert sp.simplify(expected[1]) == expected[1]


def test_cache_distinguishes_input_types():
    """Equal values of different types are cached separately."""
    _cached_to_sympy.cache_clear()
    for value in (1, "1", Fraction(1), True, sp.Integer(1)):
        assert ExactNumber(value) == 1
    assert _outcome(lambda: ExactNumber(1.0))[1] is ValueError
    _cached_to_sympy.cache_clear()
    ExactNumber(Fraction(1))
    # Only int, str and Fraction are cached; bool and float are not.
    ExactNumber(True)
    ExactNumber(float("inf"))
    ExactNumber(1)
    ExactNumber("1")
    info = _cached_to_sympy.cache_info()
    assert info.currsize == 3
    assert info.hits == 0


def test_invalid_inputs_are_rejected_every_time():
    """Errors are not cached, so invalid inputs keep raising the same error."""
    for value in ("x + 1", 3.5, "pi + I"):
        first = _outcome(partial(ExactNumber, value))
        second = _outcome(partial(ExactNumber, value))
        assert first[0] == "error"
        assert first == second


def test_exact_number_input_is_reused():
    """Constructing from an ExactNumber reuses its already-verified expression."""
    value = ExactNumber("sqrt(5/3)")
    assert ExactNumber(value).expr is value.expr


@pytest.mark.parametrize(
    "value",
    [
        sp.Integer(7),
        sp.Rational(-2, 9),
        sp.S.Zero,
        sp.oo,
        -sp.oo,
    ],
    ids=repr,
)
def test_simplified_sympy_numbers_are_unchanged(value: sp.Expr):
    """Rationals and infinities are used as is, and simplify would not change them."""
    assert ExactNumber(value).expr is value
    assert sp.simplify(value) == value
    assert type(sp.simplify(value)) is type(value)


_NON_EXPR_INPUTS = [
    sp.Eq(1, 2),
    sp.Eq(sp.symbols("x"), 2, evaluate=False),
    sp.StrictGreaterThan(1, 0, evaluate=False),
    sp.Tuple(1, 2),
]


@pytest.mark.parametrize("value", _NON_EXPR_INPUTS, ids=repr)
def test_non_expr_fails_like_type_checked_verifier(value: Any):
    """The unchecked verifier raises the same error for non-Expr values."""
    expected = _outcome(lambda: _verify_expr_is_an_exact_number(value))
    assert expected[0] == "error"
    assert _outcome(lambda: _verify_expr_recursively(value)) == expected


@pytest.mark.parametrize("value", _NON_EXPR_INPUTS, ids=repr)
def test_non_expr_rejected_without_type_checking(value: Any):
    """Non-Expr values are rejected, not recursed on, if type checking is disabled.

    Type checking is disabled with ``python -O``; this simulates it by replacing the
    type checked functions with unchecked versions and calling the unchecked
    verifier directly.
    """

    def unchecked(expr: Any) -> None:
        """Unchecked version of the type checked no-op."""

    with (
        patch.object(exact_number, "_check_is_sympy_expr", unchecked),
        patch.object(
            exact_number, "_verify_expr_is_an_exact_number", _verify_expr_recursively
        ),
    ):
        with pytest.raises(UnsupportedSympyExprError):
            _verify_expr_recursively(value)


def test_non_expr_rejected_with_python_optimize():
    """Non-Expr values raise UnsupportedSympyExprError under ``python -O``.

    ``python -O`` disables type checking, which previously led to a RecursionError.
    """
    code = textwrap.dedent(
        """
        import sympy as sp
        from tmlt.core.exceptions import UnsupportedSympyExprError
        from tmlt.core.utils.exact_number import (
            ExactNumber,
            _verify_expr_is_an_exact_number,
        )
        for func in (
            lambda: ExactNumber("1 > 0"),
            lambda: _verify_expr_is_an_exact_number(sp.Eq(1, 2)),
        ):
            try:
                func()
            except UnsupportedSympyExprError:
                continue
            raise AssertionError("expected UnsupportedSympyExprError")
        """
    )
    subprocess.run([sys.executable, "-O", "-c", code], check=True, timeout=120)
