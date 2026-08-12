"""Shared machinery for the pandas structural transformation suites.

Three things live here, all of them used by more than one of the suites for
:mod:`tmlt.core.transformations.pandas_transformations`:

* :data:`D_IN_GRID` and :func:`assert_stability_parity`, which pin a pandas
  transformation's stability function against its Spark twin's over a fixed
  grid of distances rather than against a hard-coded expectation. The two
  implementations' ``stability_function`` bodies are copies of each other, and
  this is what keeps them copies.
* :func:`pandas_domain_for_case` and :func:`spark_domain_for_case`, which
  describe an :class:`~test.unit.backend_testing.corpus.EdgeCase` as a domain
  for either backend, and :func:`describable_cases`, which is the subset of the
  corpus the pandas column descriptors can describe at all.
* :func:`labelled_value`, the backend-independent rendering of a row value that
  the differential Map tests hand to their user function.

This is deliberately *not* in :mod:`test.unit.backend_testing`: that package is
backend-neutral and frozen, and everything here knows what a
:class:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor` is.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import math
from test.unit.backend_testing import EDGE_CASES, EdgeCase
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp
from pyspark.sql.types import DateType, StringType

from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.transformations.base import Transformation

#: The distances every stability parity test is checked over: the two ends of
#: the useful range, a couple of ordinary values, a non-integer, and infinity.
D_IN_GRID: Tuple[Any, ...] = (0, 1, 2, 7, sp.Integer(3) / 2, sp.oo)


def _stability_outcome(transformation: Transformation, d_in: Any) -> Tuple[Any, ...]:
    """Returns what a transformation's stability function does with a distance.

    A distance the metric rejects -- 3/2 under
    :class:`~tmlt.core.metrics.SymmetricDifference`, say -- is as much a part of
    a stability function's behaviour as one it accepts, so the rejection is
    returned rather than raised.

    Args:
        transformation: The transformation to ask.
        d_in: The distance to ask about.
    """
    try:
        return ("value", transformation.stability_function(d_in))
    except Exception as exception:
        return ("error", type(exception).__name__, str(exception))


def assert_stability_parity(
    pandas_transformation: Transformation, spark_transformation: Transformation
) -> None:
    """Asserts two transformations' stability functions agree on :data:`D_IN_GRID`.

    They agree when they return the same distance for the same ``d_in``, and
    when they reject the same ``d_in`` with the same error.

    Args:
        pandas_transformation: The pandas transformation.
        spark_transformation: The Spark transformation it mirrors.
    """
    for d_in in D_IN_GRID:
        expected = _stability_outcome(spark_transformation, d_in)
        actual = _stability_outcome(pandas_transformation, d_in)
        assert actual == expected, (
            f"stability_function({d_in}) gives {actual} for the pandas "
            f"transformation and {expected} for its Spark twin."
        )


################################################################################
# Describing a corpus case as a domain
################################################################################


def _descriptor_for(
    field_type: Any, pandas_dtype: str
) -> Optional[PandasColumnDescriptor]:
    """Returns the descriptor for a corpus column, or None if there is none.

    Every descriptor is as permissive as it can be, since the transformations
    under test do not filter values: the point of these domains is to *describe*
    the corpus, not to constrain it.

    Args:
        field_type: The Spark type of the column, which is what tells the two
            kinds of object column apart.
        pandas_dtype: The pandas dtype of the column.
    """
    if pandas_dtype in ("int64", "int32"):
        return PandasIntegerColumnDescriptor(
            allow_null=False, size=int(pandas_dtype[3:])
        )
    if pandas_dtype in ("Int64", "Int32"):
        return PandasIntegerColumnDescriptor(
            allow_null=True, size=int(pandas_dtype[3:])
        )
    if pandas_dtype in ("float64", "float32", "Float64", "Float32"):
        return PandasFloatColumnDescriptor(
            allow_nan=True,
            allow_inf=True,
            allow_null=pandas_dtype[0].isupper(),
            size=int(pandas_dtype[5:]),
        )
    if pandas_dtype.startswith("datetime64"):
        return PandasTimestampColumnDescriptor(allow_null=True)
    if pandas_dtype == "object":
        # An object column carries no type of its own; the case's Spark type is
        # what says which kind of values it holds.
        if isinstance(field_type, StringType):
            return PandasStringColumnDescriptor(allow_null=True)
        if isinstance(field_type, DateType):
            return PandasDateColumnDescriptor(allow_null=True)
        # Binary values, and the object columns holding floats alongside nulls,
        # have no pandas descriptor.
        return None
    # pandas' own string extension dtype is deliberately not described; see
    # PandasStringColumnDescriptor.
    return None


def pandas_domain_for_case(case: EdgeCase) -> Optional[PandasTableDomain]:
    """Returns the domain of a corpus case's pandas rendering, if there is one.

    Args:
        case: The case to describe.

    Returns:
        The domain, or None if any of the case's columns has no pandas
        descriptor.
    """
    schema: Dict[str, PandasColumnDescriptor] = {}
    for field in case.spark_schema.fields:
        descriptor = _descriptor_for(field.dataType, case.pandas_dtypes[field.name])
        if descriptor is None:
            return None
        schema[field.name] = descriptor
    return PandasTableDomain(schema)


def spark_domain_for_case(case: EdgeCase) -> SparkDataFrameDomain:
    """Returns the domain of a corpus case's Spark rendering.

    It is built from the pandas domain rather than from the case's Spark schema,
    so that the two backends' transformations under test are given the *same*
    description of the same data.

    Args:
        case: The case to describe.

    Raises:
        ValueError: If the case has no pandas domain.
    """
    pandas_domain = pandas_domain_for_case(case)
    if pandas_domain is None:
        raise ValueError(f"Case {case.id} cannot be described by pandas descriptors.")
    return SparkDataFrameDomain(
        {
            column: descriptor.to_spark_descriptor()
            for column, descriptor in pandas_domain.schema.items()
        }
    )


def describable_cases() -> List[EdgeCase]:
    """Returns the corpus cases the pandas column descriptors can describe.

    The rest -- binary columns, pandas' string extension dtype, and the object
    columns holding floats -- are outside what
    :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain` can describe,
    so no transformation over them can be built in the first place.
    """
    return [case for case in EDGE_CASES if pandas_domain_for_case(case) is not None]


################################################################################
# Rendering row values
################################################################################


def labelled_value(value: Any) -> str:
    """Returns a string rendering of a row value, identical on both backends.

    This is what the differential Map suites' user function is built out of. It
    distinguishes everything the two backends should agree on -- a missing value
    from a NaN, an int from a float, ``0.0`` from ``-0.0``, a date from a
    timestamp -- and renders each with a Python builtin, which is the same
    function in the pandas process and in a Spark executor. Timestamps are the
    one value the two hand over as different *types*
    (:class:`pandas.Timestamp` against :class:`datetime.datetime`), and
    ``isoformat`` renders both identically.

    Args:
        value: The value to render.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"float:{value!r}:{math.copysign(1.0, value):.0f}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes:{bytes(value).hex()}"
    if isinstance(value, datetime.datetime):
        return f"timestamp:{value.isoformat()}"
    if isinstance(value, datetime.date):
        return f"date:{value.isoformat()}"
    return f"other:{value!r}"
