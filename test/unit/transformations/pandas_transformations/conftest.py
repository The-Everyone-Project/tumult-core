"""Shared fixtures for the pandas transformation suites."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.pandas_metric_bridge import pandas_metric_support
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _pandas_metrics() -> Iterator[None]:
    """Gives the non-grouped metrics pandas support for every test here.

    A pandas transformation cannot be constructed at all until the metrics know
    about pandas tables; see :mod:`test.unit.pandas_metric_bridge` for why that
    is bridged here rather than implemented.

    Yields:
        Nothing.
    """
    with pandas_metric_support():
        yield
