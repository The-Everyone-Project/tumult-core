"""Unit tests for :mod:`tmlt.core.utils.validation`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import itertools
from fractions import Fraction
from typing import Any, Callable, Optional, Tuple
from unittest.mock import patch

import pytest
import sympy as sp

from tmlt.core.utils import validation
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.validation import validate_exact_number

_VALUES = [
    0,
    1,
    2,
    -1,
    -2,
    10**30,
    True,
    False,
    Fraction(1, 2),
    Fraction(2, 1),
    "3",
    "0.5",
    sp.Integer(1),
    sp.Rational(3, 2),
    sp.oo,
    -sp.oo,
    float("inf"),
    3.5,
    ExactNumber(1),
    "x + 1",
    None,
]

_BOUNDS = [None, 0, 1, -1, True, Fraction(1, 2), "1", ExactNumber(1), sp.oo]


def _outcome(func: Callable[[], Any]) -> Tuple[Any, ...]:
    """Returns ("ok", result) or ("error", exception type, message) for func()."""
    try:
        return ("ok", func())
    except Exception as e:
        return ("error", type(e), str(e))


@pytest.mark.parametrize(
    "allow_nonintegral,minimum_is_inclusive,maximum_is_inclusive",
    list(itertools.product([True, False], repeat=3)),
)
@pytest.mark.parametrize("value", _VALUES, ids=repr)
def test_validate_exact_number_fast_path_matches_slow_path(
    value: Any,
    allow_nonintegral: bool,
    minimum_is_inclusive: bool,
    maximum_is_inclusive: bool,
):
    """The plain-int fast path accepts and rejects exactly what the full check does."""
    for minimum, maximum in itertools.product(_BOUNDS, repeat=2):

        def check(
            minimum: Optional[Any] = minimum, maximum: Optional[Any] = maximum
        ) -> None:
            validate_exact_number(
                value,
                allow_nonintegral=allow_nonintegral,
                minimum=minimum,
                minimum_is_inclusive=minimum_is_inclusive,
                maximum=maximum,
                maximum_is_inclusive=maximum_is_inclusive,
            )

        with patch.object(validation, "_plain_int_is_valid", return_value=False):
            expected = _outcome(check)
        assert _outcome(check) == expected, (minimum, maximum)
