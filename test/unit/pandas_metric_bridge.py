"""A temporary stand-in for the pandas branches of the non-grouped metrics.

The grouped pandas stack -- the ``PandasGroupedTable`` in
:mod:`tmlt.core.utils.pandas_grouped_table`,
:class:`~tmlt.core.domains.pandas_domains.PandasGroupedTableDomain`, and the
transformations in
:mod:`tmlt.core.transformations.pandas_transformations` -- cannot be exercised
without the *non*-grouped metrics knowing about pandas tables:

* :class:`~tmlt.core.transformations.base.Transformation` rejects any
  domain/metric pair the metric says it does not support, so a pandas
  ``GroupBy`` cannot even be constructed unless
  :class:`~tmlt.core.metrics.SymmetricDifference` supports
  :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`, and a pandas
  ``CountGrouped`` cannot unless :class:`~tmlt.core.metrics.OnColumn` does.
* :meth:`~tmlt.core.metrics.AggregationMetric.distance` over a grouped domain
  delegates to its inner metric over the *group* domain, which is a
  ``PandasTableDomain``.

Those branches belong to a parallel work package, which owns
:class:`~tmlt.core.metrics.SymmetricDifference`,
:class:`~tmlt.core.metrics.HammingDistance`,
:class:`~tmlt.core.metrics.OnColumn` and
:class:`~tmlt.core.metrics.AddRemoveKeys` and has not landed yet. Rather than
edit those classes here -- which would collide with it -- this module supplies
the three the grouped stack needs, for the duration of a test, and supplies
*nothing* once the real implementations are there:
:data:`BRIDGE_NEEDED` is False then, and :func:`pandas_metric_support` becomes a
no-op. Deleting this module once that is permanently the case is the intended
end state.

The stand-in semantics are deliberately the obvious ones -- a multiset
difference under
:func:`~tmlt.core.utils.pandas_grouping.row_keys`' notion of row identity, and
the grouped rules copied from the Spark branch -- so that a test passing here is
a test of the grouped code, not of this file. Nothing in ``src`` imports it.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from collections import Counter
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pandas as pd
import sympy as sp

from tmlt.core.domains.base import Domain
from tmlt.core.domains.pandas_domains import (
    PandasGroupedTableDomain,
    PandasSeriesDomain,
    PandasTableDomain,
)
from tmlt.core.metrics import HammingDistance, OnColumn, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.pandas_grouping import row_keys

_ORIGINAL_SYMMETRIC_DIFFERENCE_SUPPORTS = SymmetricDifference.supports_domain
_ORIGINAL_SYMMETRIC_DIFFERENCE_DISTANCE = SymmetricDifference.distance
_ORIGINAL_HAMMING_SUPPORTS = HammingDistance.supports_domain
_ORIGINAL_ON_COLUMN_SUPPORTS = OnColumn.supports_domain

BRIDGE_NEEDED = not _ORIGINAL_SYMMETRIC_DIFFERENCE_SUPPORTS(
    SymmetricDifference(), PandasTableDomain({})
)
"""Whether the real metrics still lack pandas support, and this file is needed."""


def _multiset_distance(value1: pd.DataFrame, value2: pd.DataFrame) -> int:
    """Returns the number of rows in exactly one of two tables.

    Rows are compared with :func:`~tmlt.core.utils.pandas_grouping.row_keys`, so
    a null and a NaN are different rows, as they are in Spark.

    Args:
        value1: The first table.
        value2: The second table.
    """
    counts1 = Counter(row_keys(value1))
    counts2 = Counter(row_keys(value2))
    return sum((counts1 - counts2).values()) + sum((counts2 - counts1).values())


def _symmetric_difference_supports_domain(
    self: SymmetricDifference, domain: Domain
) -> bool:
    """Returns whether the metric supports a domain, pandas tables included."""
    if isinstance(domain, (PandasTableDomain, PandasGroupedTableDomain)):
        return True
    return _ORIGINAL_SYMMETRIC_DIFFERENCE_SUPPORTS(self, domain)


def _symmetric_difference_distance(
    self: SymmetricDifference, value1: Any, value2: Any, domain: Domain
) -> ExactNumber:
    """Returns the distance between two elements, pandas tables included."""
    if isinstance(domain, PandasTableDomain):
        self._validate_distance_arguments(value1, value2, domain)
        distance = ExactNumber(_multiset_distance(value1, value2))
        self.validate(distance)
        return distance
    if isinstance(domain, PandasGroupedTableDomain):
        self._validate_distance_arguments(value1, value2, domain)
        groups1 = value1.get_groups()
        groups2 = value2.get_groups()
        if groups1.keys() != groups2.keys():
            return ExactNumber(sp.oo)
        group_domain = domain.get_group_domain()
        distance = ExactNumber(0)
        for key in groups1:
            df1 = groups1[key]
            df2 = groups2[key]
            if self.distance(df1, df2, group_domain) > 0:
                if len(df1) == 0 or len(df2) == 0:
                    distance += 1
                else:
                    distance += 2
        return distance
    return _ORIGINAL_SYMMETRIC_DIFFERENCE_DISTANCE(self, value1, value2, domain)


def _hamming_supports_domain(self: HammingDistance, domain: Domain) -> bool:
    """Returns whether the metric supports a domain, pandas tables included."""
    if isinstance(domain, PandasTableDomain):
        return True
    return _ORIGINAL_HAMMING_SUPPORTS(self, domain)


def _on_column_supports_domain(self: OnColumn, domain: Domain) -> bool:
    """Returns whether the metric supports a domain, pandas tables included."""
    if isinstance(domain, PandasTableDomain):
        return self.column in domain.schema and self.metric.supports_domain(
            PandasSeriesDomain(domain[self.column].to_numpy_domain())
        )
    return _ORIGINAL_ON_COLUMN_SUPPORTS(self, domain)


@contextmanager
def pandas_metric_support() -> Iterator[None]:
    """Gives the non-grouped metrics pandas support, if they do not have it.

    Yields:
        Nothing; the support is in place for the duration of the ``with`` block.
    """
    if not BRIDGE_NEEDED:
        yield
        return
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                SymmetricDifference,
                "supports_domain",
                _symmetric_difference_supports_domain,
            )
        )
        stack.enter_context(
            patch.object(
                SymmetricDifference, "distance", _symmetric_difference_distance
            )
        )
        stack.enter_context(
            patch.object(HammingDistance, "supports_domain", _hamming_supports_domain)
        )
        stack.enter_context(
            patch.object(OnColumn, "supports_domain", _on_column_supports_domain)
        )
        yield
