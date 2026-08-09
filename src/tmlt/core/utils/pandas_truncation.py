"""Functions for truncating pandas DataFrames.

These functions are pandas counterparts of the Spark truncation utilities in
:mod:`tmlt.core.utils.truncation`. They implement the same algorithms, using the
same SHA-256 based row ordering, so that for every column type supported by both
backends the two implementations keep exactly the same rows.

Compatibility with :mod:`tmlt.core.utils.truncation`:
    Supported dtypes
        A column is hashable by these functions if its dtype is one of:

        * any numpy or pandas nullable integer dtype (``int8`` through
          ``int64``, ``uint8`` through ``uint64``, ``Int8`` through ``UInt64``),
          which corresponds to Spark's ``IntegerType`` and ``LongType``;
        * ``float32`` or ``Float32`` (Spark ``FloatType``), and ``float64`` or
          ``Float64`` (Spark ``DoubleType``);
        * ``datetime64[ns]`` without a timezone (Spark ``TimestampType``);
        * ``object`` and the pandas string dtypes, whose values may be
          :class:`str` (Spark ``StringType``), :class:`bytes` or
          :class:`bytearray` (Spark ``BinaryType``),
          :class:`datetime.date` (Spark ``DateType``),
          :class:`datetime.datetime` (Spark ``TimestampType``), any of the
          numeric types above, or a null value.

        Every other dtype raises :class:`NotImplementedError`, as do boolean
        columns and unsupported values inside ``object`` columns. Because an
        empty ``object`` column carries no values, unsupported *value* types
        cannot be detected in that case.

        Which columns are hashed differs by function, and so does when an
        unsupported dtype is reported: :func:`truncate_large_groups` hashes
        every column, :func:`limit_keys_per_group` hashes only the grouping and
        key columns, and :func:`drop_large_groups` hashes nothing and therefore
        never raises :class:`NotImplementedError`.

    Nulls and NaNs in float columns
        Spark distinguishes ``NULL`` from ``NaN``, and hashes them differently.
        A numpy ``float32``/``float64`` column cannot represent ``NULL``, so
        ``NaN`` values in such columns are hashed the way Spark hashes ``NaN``.
        To express ``NULL`` in a float column, use the pandas nullable
        ``Float32``/``Float64`` dtypes with ``pd.NA``. An ``object`` column can
        hold both. The two are also different group keys, as they are in Spark,
        even though a pandas ``groupby`` would put them in the same group.

    Dates and timestamps
        Timestamps are hashed and grouped as their wall-clock value, with
        sub-microsecond precision discarded, so they hash identically to Spark
        timestamps whenever the naive pandas values represent wall clocks in
        Spark's session timezone (``spark.sql.session.timeZone``). Timezone-aware
        columns raise :class:`NotImplementedError`; convert them with
        :meth:`~pandas.Series.dt.tz_convert` followed by
        :meth:`~pandas.Series.dt.tz_localize` first.

    Floating-point rendering and the JVM version
        Spark renders ``float`` and ``double`` values with the JVM's
        ``Float.toString``/``Double.toString``, and hashes the result. This
        module reimplements the rendering specified by Java 19 and later, which
        is the shortest decimal that round-trips to the value. Java 18 and
        earlier sometimes render the same value differently, usually with more
        digits than are needed (`JDK-4511638
        <https://bugs.openjdk.org/browse/JDK-4511638>`_). Those renderings
        denote the same value but hash differently, so against a Spark running
        on such a JVM some values hash differently here. Sampling uniformly
        over bit patterns, this affects roughly 0.2% of ``double`` values and
        roughly 10% of ``float`` values -- those needing many significant
        digits, plus the smallest subnormals. Java 19 and later are unaffected.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import hashlib
import math
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Collection, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from tmlt.core.utils.misc import get_nonconflicting_string

_SUPPORTED_FLOAT_DTYPES = (np.dtype("float32"), np.dtype("float64"))

_UNSUPPORTED_EXTENSION_DTYPES = (
    pd.CategoricalDtype,
    pd.IntervalDtype,
    pd.PeriodDtype,
    pd.SparseDtype,
)

# The three classes of value Spark's ascending order puts in this order: nulls
# first, then every ordinary value, then NaNs.
_NULL_ORDER = 0
_VALUE_ORDER = 1
_NAN_ORDER = 2


def _layout_java_decimal(sign: str, digits: str, decimal_exponent: int) -> str:
    """Returns the Java rendering of ``sign`` ``0.<digits>`` * 10 ** exponent."""
    # Java uses plain notation for magnitudes in [1e-3, 1e7), and computerized
    # scientific notation everywhere else.
    if -2 <= decimal_exponent <= 7:
        if decimal_exponent <= 0:
            return sign + "0." + "0" * (-decimal_exponent) + digits
        if decimal_exponent >= len(digits):
            padding = "0" * (decimal_exponent - len(digits))
            return sign + digits + padding + ".0"
        return sign + digits[:decimal_exponent] + "." + digits[decimal_exponent:]
    mantissa = digits[0] + "." + (digits[1:] or "0")
    return sign + mantissa + "E" + str(decimal_exponent - 1)


def _shortest_digits(text: str) -> Tuple[str, int]:
    """Returns the significant digits and decimal exponent of a decimal string.

    The value is ``0.<digits> * 10 ** exponent``. Trailing zeros are stripped
    because they are not significant: for example ``repr(5152716558868863.0)``
    is ``'5152716558868863.0'``, whose final zero must not be counted.
    """
    as_tuple = Decimal(text).as_tuple()
    digits = "".join(map(str, as_tuple.digits)).rstrip("0") or "0"
    return digits, int(as_tuple.exponent) + len(as_tuple.digits)


def _two_significant_digits(exact: Decimal) -> Tuple[str, int]:
    """Returns the two-digit decimal closest to ``exact``, and its exponent.

    Java picks the shortest decimal that round-trips, except that when a single
    digit suffices it instead picks whichever decimal with one or two digits is
    closest to the exact value, breaking ties towards an even last digit. This
    only ever changes the result for tiny subnormal values, where the gap
    between adjacent floating point numbers is large relative to their
    magnitude: ``Double.MIN_VALUE`` renders as ``4.9E-324``, not ``5.0E-324``.
    """
    decimal_exponent = exact.adjusted() + 1
    scaled = exact.scaleb(2 - decimal_exponent)
    significand = int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
    if significand >= 100:
        significand //= 10
        decimal_exponent += 1
    return (str(significand).rstrip("0") or "0"), decimal_exponent


def _java_double_to_string(value: float) -> str:
    """Returns the value as Java's ``Double.toString`` renders it.

    The argument must be finite: infinities and NaNs are special-cased by
    :func:`_render_value` before it reaches this function, exactly as the Spark
    implementation special-cases them before casting to a string.
    """
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    digits, decimal_exponent = _shortest_digits(repr(magnitude))
    if len(digits) == 1:
        digits, decimal_exponent = _two_significant_digits(Decimal(magnitude))
    return _layout_java_decimal(sign, digits, decimal_exponent)


def _java_float_to_string(value: np.float32) -> str:
    """Returns the value as Java's ``Float.toString`` renders it.

    The argument must be finite, for the same reason as in
    :func:`_java_double_to_string`.
    """
    if value == np.float32(0.0):
        return "-0.0" if math.copysign(1.0, float(value)) < 0 else "0.0"
    sign = "-" if value < np.float32(0.0) else ""
    magnitude = np.float32(abs(float(value)))
    digits, decimal_exponent = _shortest_digits(
        np.format_float_positional(magnitude, unique=True, trim="-")
    )
    if len(digits) == 1:
        digits, decimal_exponent = _two_significant_digits(Decimal(float(magnitude)))
    return _layout_java_decimal(sign, digits, decimal_exponent)


def _sha256(data: bytes) -> str:
    """Returns the hex-encoded SHA-256 digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def _is_null(value: Any) -> bool:
    """Returns whether a value is a null value, as opposed to a float NaN."""
    return value is None or value is pd.NA or value is pd.NaT


def _render_value(value: Any) -> Optional[bytes]:
    """Renders a single value as the bytes Spark hashes for it.

    Returns:
        The rendering Spark's ``_hash_column`` would hash, or None if the value
        is null.
    """
    if _is_null(value):
        return None
    # isinstance(True, int) is True, so booleans must be rejected before the
    # integer branch is reached.
    if isinstance(value, (bool, np.bool_)):
        raise NotImplementedError("Unsupported data type bool")
    if isinstance(value, np.float32):
        # np.float32 is not a subclass of float, so it must be dispatched
        # before the general float branch.
        if np.isnan(value):
            return b"nan"
        if np.isinf(value):
            return b"-inf" if value < 0 else b"inf"
        return _java_float_to_string(value).encode("utf-8")
    if isinstance(value, (float, np.float64)):
        if math.isnan(value):
            return b"nan"
        if math.isinf(value):
            return b"-inf" if value < 0 else b"inf"
        return _java_double_to_string(float(value)).encode("utf-8")
    if isinstance(value, (int, np.integer)):
        return str(int(value)).encode("utf-8")
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    # datetime.datetime is a subclass of datetime.date, so it must be
    # dispatched first.
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            raise NotImplementedError(
                "Unsupported data type timezone-aware datetime; convert "
                "timestamps to wall-clock values first, for example with "
                "series.dt.tz_convert('UTC').dt.tz_localize(None)"
            )
        rendered = (
            f"{value.year:04d}-{value.month:02d}-{value.day:02d} "
            f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        )
        # Spark prints as many fractional digits as are needed, and none at all
        # when the fraction is zero. Anything finer than a microsecond is
        # discarded.
        if value.microsecond:
            rendered += "." + f"{value.microsecond:06d}".rstrip("0")
        return rendered.encode("utf-8")
    if isinstance(value, datetime.date):
        return value.isoformat().encode("utf-8")
    raise NotImplementedError(f"Unsupported data type {type(value).__name__}")


def _hash_value(value: Any) -> Optional[str]:
    """Hashes a single value the way Spark's ``_hash_column`` hashes it.

    Returns:
        The hex-encoded SHA-256 digest of the value's Spark string rendering,
        or None if the value is null.
    """
    rendered = _render_value(value)
    return None if rendered is None else _sha256(rendered)


def _combined_hash(values: Sequence[Any]) -> str:
    """Combines the hashes of ``values`` into a single hash.

    This mirrors Spark's ``_hash_columns``: the per-value hashes are joined
    with commas, skipping nulls, and that string is hashed, and its digest
    hashed once more.

    Returns:
        The hex-encoded SHA-256 digest for the given values.
    """
    hashes = [hashed for hashed in map(_hash_value, values) if hashed is not None]
    concatenated = _sha256(",".join(hashes).encode("utf-8"))
    return _sha256(concatenated.encode("utf-8"))


def _column_values(column: pd.Series) -> Iterator[Any]:
    """Returns the values of a column, with the precision of its dtype."""
    if column.dtype == np.dtype("float32"):
        # Iterating a numpy float32 series yields Python floats, which are
        # double precision and would be rendered with too many digits.
        return iter(column.to_numpy(dtype=np.float32))
    if isinstance(column.dtype, pd.Float32Dtype):
        return iter(column.array)
    return iter(column)


def _validate_column(column: pd.Series, name: str) -> None:
    """Raises an error if the column has a dtype that cannot be hashed."""
    dtype = column.dtype
    message = f"Unsupported data type {dtype} for column {name}"
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise NotImplementedError(
            f"{message}; convert timestamps to wall-clock values first, for "
            "example with series.dt.tz_convert('UTC').dt.tz_localize(None)"
        )
    # These have to be rejected up front: a categorical dtype whose categories
    # are integers, for example, passes the integer check below.
    if isinstance(dtype, _UNSUPPORTED_EXTENSION_DTYPES):
        raise NotImplementedError(message)
    if pd.api.types.is_bool_dtype(dtype):
        raise NotImplementedError(message)
    if pd.api.types.is_integer_dtype(dtype):
        return
    if dtype in _SUPPORTED_FLOAT_DTYPES or isinstance(
        dtype, (pd.Float32Dtype, pd.Float64Dtype)
    ):
        return
    if pd.api.types.is_datetime64_dtype(dtype):
        return
    if isinstance(dtype, pd.StringDtype):
        return
    if pd.api.types.is_object_dtype(dtype):
        # An object column has no type of its own, so every value has to be
        # checked. An empty one carries no values and so cannot be checked,
        # unlike the Spark schema it corresponds to.
        for value in _column_values(column):
            _render_value(value)
        return
    raise NotImplementedError(message)


def _hash_columns(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Hashes the given columns of every row into a single value.

    Returns:
        A series of hex-encoded SHA-256 digests, aligned with ``df``.
    """
    for column in columns:
        _validate_column(df[column], column)
    if columns:
        values = [_column_values(df[column]) for column in columns]
        hashes = [_combined_hash(row) for row in zip(*values)]
    else:
        hashes = [_combined_hash(())] * len(df)
    return pd.Series(hashes, index=df.index, dtype=object)


def _group_key(value: Any) -> Tuple[int, Any]:
    """Returns the key Spark groups and orders a value by.

    Spark's window partitioning and ordering differ from what a pandas
    ``groupby`` or ``sort_values`` does in four ways, all of which this key
    encodes:

    * A null and a NaN are different partitions, and ascending order puts nulls
      first and NaNs last, while pandas puts both in the same group and, with
      ``na_position``, in the same place. This is reachable in an ``object``
      column, which can hold both.
    * ``-0.0`` and ``0.0`` are one partition, and tie in an ordering, even
      though they hash differently. Two Python floats already behave that way.
    * Binary values are compared by content, and a ``bytearray`` is not even
      hashable, so binary values are keyed by their bytes.
    * Timestamps have microsecond resolution. Values are hashed with
      sub-microsecond precision discarded, so grouping and ordering have to
      discard it too, or a ``datetime64[ns]`` column would split a Spark
      partition in two.

    Returns:
        A hashable key whose natural order is Spark's ascending order.
    """
    if _is_null(value):
        return (_NULL_ORDER, 0)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return (_NAN_ORDER, 0)
        return (_VALUE_ORDER, float(value))
    if isinstance(value, (bytes, bytearray)):
        return (_VALUE_ORDER, bytes(value))
    if isinstance(value, pd.Timestamp) and value.nanosecond:
        return (_VALUE_ORDER, value.floor("us"))
    return (_VALUE_ORDER, value)


def _group_keys(column: pd.Series) -> pd.Series:
    """Returns the keys grouping a column the way Spark groups it.

    Returns:
        A series of :func:`_group_key` values, aligned with ``column``.
    """
    keys = [_group_key(value) for value in _column_values(column)]
    return pd.Series(keys, index=column.index, dtype=object)


def _sorted_keys(keys: Set[Tuple[int, Any]]) -> List[Tuple[int, Any]]:
    """Returns a column's group keys in Spark's ascending order.

    Returns:
        The keys, sorted.
    """
    try:
        return sorted(keys)
    except TypeError:
        # A column holding values of several types has no Spark counterpart,
        # since a Spark column has a single type. Falling back to ordering such
        # values by type name keeps the sort deterministic rather than failing.
        return sorted(keys, key=lambda key: (key[0], type(key[1]).__name__, key[1]))


def _order_codes(column: pd.Series) -> np.ndarray:
    """Returns integer codes ordering a column the way Spark orders it.

    Returns:
        One code per row, in the column's order, ordered like
        :func:`_group_key`.
    """
    keys = [_group_key(value) for value in _column_values(column)]
    ranks = {key: rank for rank, key in enumerate(_sorted_keys(set(keys)))}
    return np.array([ranks[key] for key in keys], dtype=np.int64)


def _sort_by(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Sorts ``df`` by ``columns`` in ascending order, as Spark orders rows.

    The rows are ordered by :func:`_order_codes` rather than by the values
    themselves, because no ``na_position`` reproduces Spark's ordering of nulls
    and NaNs. ``columns`` must not be empty.

    Returns:
        A copy of ``df`` with its rows sorted, keeping the index labels.
    """
    # numpy's lexsort takes the last key as the primary one, and is stable.
    codes = [_order_codes(df[column]) for column in reversed(columns)]
    return df.iloc[np.lexsort(codes)]


def _group_by(df: pd.DataFrame, columns: Collection[str]) -> Any:
    """Groups ``df`` by ``columns``, treating no columns as a single group.

    The rows are grouped by :func:`_group_key` rather than by the values
    themselves, so that the groups are the partitions Spark would form. Groups
    are not sorted.

    Returns:
        The pandas groupby object for the given columns.
    """
    if columns:
        # Spark accepts a repeated partitioning column; grouping by the same
        # column twice here would only be wasted work.
        unique_columns = list(dict.fromkeys(columns))
        # The keys are never null -- a null value is a key of its own -- so
        # dropna only matters in that it keeps pandas from looking for nulls.
        keys = [_group_keys(df[column]) for column in unique_columns]
        return df.groupby(keys, sort=False, dropna=False)
    return df.groupby(np.zeros(len(df), dtype=np.int64), sort=False, dropna=False)


def truncate_large_groups(
    df: pd.DataFrame, grouping_columns: Collection[str], threshold: int
) -> pd.DataFrame:
    """Order rows by a hash function and keep at most ``threshold`` rows for each group.

    This is the pandas counterpart of
    :func:`tmlt.core.utils.truncation.truncate_large_groups`, and keeps the same
    rows as that function does.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2", "b3"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(truncate_large_groups(dataframe, ["A"], 3))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(truncate_large_groups(dataframe, ["A"], 2))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> print_pandas(truncate_large_groups(dataframe, ["A"], 1))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        threshold: Maximum number of rows to include for each group.
    """
    starting_columns = list(df.columns)
    working_df = df.reset_index(drop=True)
    for column in starting_columns:
        _validate_column(working_df[column], column)
    # Identical rows must hash differently, or they would be kept or dropped as
    # a block. Spark numbers them with row_number over a window partitioned by
    # every column, which is a cumulative count over identical rows.
    row_index_column = get_nonconflicting_string(starting_columns)
    working_df[row_index_column] = (
        _group_by(working_df, starting_columns).cumcount() + 1
    )
    hashed_columns = [*starting_columns, row_index_column]
    hash_column = get_nonconflicting_string(hashed_columns)
    working_df[hash_column] = _hash_columns(working_df, hashed_columns)
    working_df = _sort_by(working_df, [hash_column, *starting_columns])
    rank = _group_by(working_df, grouping_columns).cumcount() + 1
    truncated = working_df.loc[rank <= threshold, starting_columns]
    return truncated.reset_index(drop=True)


def drop_large_groups(
    df: pd.DataFrame, grouping_columns: List[str], threshold: int
) -> pd.DataFrame:
    """Drop all rows for groups that have more than ``threshold`` rows.

    This is the pandas counterpart of
    :func:`tmlt.core.utils.truncation.drop_large_groups`, and keeps the same
    rows as that function does. It does not hash any values, so it never raises
    an error for unsupported column types.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2", "b3"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(drop_large_groups(dataframe, ["A"], 3))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(drop_large_groups(dataframe, ["A"], 2))
            A   B
        0  a1  b1
        1  a2  b1
        >>> print_pandas(drop_large_groups(dataframe, ["A"], 1))
            A   B
        0  a1  b1
        1  a2  b1

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        threshold: Threshold for dropping groups. If more than ``threshold`` rows belong
            to the same group, all rows in that group are dropped.
    """
    starting_columns = list(df.columns)
    count_column = get_nonconflicting_string(starting_columns)
    working_df = df.reset_index(drop=True)
    group_ids = _group_by(working_df, grouping_columns).ngroup()
    working_df[count_column] = group_ids.map(group_ids.value_counts())
    kept = working_df.loc[working_df[count_column] <= threshold, starting_columns]
    return kept.reset_index(drop=True)


def limit_keys_per_group(
    df: pd.DataFrame,
    grouping_columns: Collection[str],
    key_columns: Collection[str],
    threshold: int,
) -> pd.DataFrame:
    """Order keys by a hash function and keep at most ``threshold`` keys for each group.

    This is the pandas counterpart of
    :func:`tmlt.core.utils.truncation.limit_keys_per_group`, and keeps the same
    rows as that function does.

    .. note::

        After truncation there may still be an unbounded number of rows per key, but
        at most ``threshold`` keys per group

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4"],
            ...         "B": ["b1", "b1", "b2", "b2", "b3", "b1", "b2", "b3"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        5  a4  b1
        6  a4  b2
        7  a4  b3
        >>> print_pandas(
        ...     limit_keys_per_group(
        ...         df=dataframe,
        ...         grouping_columns=["A"],
        ...         key_columns=["B"],
        ...         threshold=2,
        ...     )
        ... )
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        5  a4  b2
        6  a4  b3
        >>> print_pandas(
        ...     limit_keys_per_group(
        ...         df=dataframe,
        ...         grouping_columns=["A"],
        ...         key_columns=["B"],
        ...         threshold=1,
        ...     )
        ... )
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b3
        3  a4  b3

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        key_columns: Column defining the keys.
        threshold: Maximum number of keys to include for each group.
    """
    starting_columns = list(df.columns)
    working_df = df.reset_index(drop=True)
    hash_column = get_nonconflicting_string(starting_columns)
    working_df[hash_column] = _hash_columns(
        working_df, [*grouping_columns, *key_columns]
    )
    # The hash only depends on the grouping and key columns, so all rows of a
    # (group, key) pair share it. Spark ranks the pairs with dense_rank; here
    # each pair is given an id, ranked once, and the surviving ids select rows.
    # Spark's dense_rank ranks by (hash, *key_columns), so the hash is part of
    # the pair's identity: pandas considers -0.0 and 0.0 equal keys, but they
    # hash differently and Spark counts them as two keys.
    pair_column = get_nonconflicting_string([*starting_columns, hash_column])
    working_df[pair_column] = _group_by(
        working_df, [*grouping_columns, hash_column, *key_columns]
    ).ngroup()
    pairs = working_df.drop_duplicates(subset=[pair_column])
    pairs = _sort_by(pairs, [hash_column, *key_columns])
    rank = _group_by(pairs, grouping_columns).cumcount() + 1
    # The pair ids are 0, 1, ... and so index a mask of the surviving pairs
    # directly, which selects the rows belonging to those pairs.
    surviving = np.zeros(len(pairs), dtype=bool)
    surviving[pairs.loc[rank <= threshold, pair_column].to_numpy(dtype=np.int64)] = True
    kept = working_df.loc[surviving[working_df[pair_column].to_numpy(dtype=np.int64)]]
    return kept[starting_columns].reset_index(drop=True)
