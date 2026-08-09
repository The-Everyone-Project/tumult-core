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
        * ``datetime64`` without a timezone (Spark ``TimestampType``), in the
          ``ns`` unit or, on pandas 2, any of the coarser ``s``/``ms``/``us``
          units -- whose values may lie far outside the ``ns`` range and are
          hashed at their own precision, never through a narrowing cast;
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

Row order:
    All three functions return their surviving rows in input order: a
    surviving row precedes another in the result exactly when it did in the
    input. The hash order decides only *which* rows survive, never how they
    are returned. (Spark makes no ordering promise at all; its output order
    is whatever the shuffle produced.)

Performance:
    Values are rendered and hashed once per *distinct* value of a column, so
    hashing cost scales with column cardinality rather than row count. The
    exception is floating point columns, whose Java rendering has no
    vectorized equivalent: a float column costs one rendering per distinct
    bit pattern, which for a near-all-distinct million-row float column is
    seconds, not milliseconds.

Fast paths:
    Both hashing functions restrict hashing and sorting to the rows that can
    actually be truncated. Let ``G`` be the set of groups whose size exceeds
    the threshold, and let ``S`` be the rows belonging to a group in ``G``.
    The fast path computes the truncation on ``S`` alone and keeps every row
    outside ``S``. This is exact:

    1. Rows outside ``S`` all survive the full path. A group of size
       ``m <= threshold`` contributes its first ``min(m, threshold) = m``
       rows in the hash order -- i.e. all of them -- regardless of what that
       order is. So restricting attention to ``S`` cannot change their fate.
    2. ``S`` is a union of whole groups, by construction.
    3. The duplicate-row salt is group-local (the salt-locality step). The
       salt is a cumulative count over a partition of *every* column. Two
       rows in the same all-columns partition agree on every column, and the
       grouping columns are a subset of the frame's columns (they must be,
       or indexing raises), so they agree on the grouping columns and lie in
       the same group. Each all-columns partition is therefore contained in
       a single group, which by (2) is either wholly inside ``S`` or wholly
       outside it. The count follows frame order, and taking a subsequence
       preserves relative order, so every row in ``S`` gets the same salt it
       would have got from the full frame. The partition itself is intrinsic
       to the values, so computing it on ``S``'s rows directly gives the
       same partition as restricting a full-frame computation.
    4. The digest depends only on a row's own values and its salt, both
       unchanged by (3).
    5. The order and the grouping are computed on the full frame and then
       restricted (the restriction step). The order keys and the grouping
       columns' codes are evaluated over all rows *before* ``S`` is chosen,
       and the resulting arrays are indexed with ``S``'s positions. The keys
       are therefore literally the same numbers the full path would compare,
       so the order induced on ``S`` is the full path's order restricted to
       ``S``, and the stable sort breaks ties by position in ``S``, which
       preserves relative input order exactly as it does in the full path.
       In particular this holds even for mixed-type object columns, where
       the ordering falls back to a type-name key: that fallback is also
       decided once, over the whole frame.
    6. Rank is a within-group prefix. For a group ``g`` in ``G``, the rows of
       ``g`` in the restricted order are exactly the rows of ``g`` in the
       full order, so the first ``threshold`` of them are the same rows.

    Hence the surviving multiset is identical. With rows returned in input
    order, the surviving *frame* is identical, which is what
    ``test_fast_path_matches_full_path`` asserts.

    For :func:`limit_keys_per_group` the same argument applies with "rank"
    replaced by ``dense_rank`` over distinct (group, key) pairs, plus one
    extra step: the budget test uses a *refinement* of Spark's pair identity,
    so the per-group pair count is an over-estimate. An over-estimate can
    only put a group *into* ``G`` that did not need to be there (correct,
    just slower); it can never leave a group out, because refined count
    ``>=`` true count means refined count ``<= threshold`` implies true count
    ``<= threshold``, and such a group keeps all of its keys and hence all of
    its rows.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import hashlib
import math
from decimal import ROUND_HALF_EVEN, Decimal
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd

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

#: Code marking a null value in a digest-code array. A null has no digest and
#: is skipped by the combiner, so it cannot share a code with any real value.
#: It coincides with the sentinel ``pd.factorize`` uses for missing values.
_NULL_DIGEST_CODE = -1

#: Object-column kinds, as reported by
#: ``pandas.api.types.infer_dtype(skipna=True)``, whose values are all
#: renderable and all faithfully factorized by ``pd.factorize``. Kinds like
#: ``mixed`` (which covers bytearrays, unhashable by ``pd.factorize``) and
#: ``mixed-integer-float`` (where ``pd.factorize`` merges ``1`` with ``1.0``,
#: which render differently) are deliberately absent.
_HOMOGENEOUS_OBJECT_KINDS = frozenset({"string", "bytes", "empty"})

#: Object-column kinds whose non-NA values are all renderable, so that
#: validation needs no per-value scan (the NA-like values, which the
#: ``skipna=True`` kind inference cannot see, are always checked separately).
#: ``datetime`` is deliberately absent, because a timezone-aware datetime must
#: still be rejected, and so is ``date``, which also covers columns mixing
#: dates with (possibly timezone-aware) datetimes; both keep a scan, at one
#: rendering per distinct value. ``floating`` also covers ``np.float16`` and
#: ``np.longdouble`` values that have no Spark rendering, and equal floats of
#: different widths may differ in renderability, so it keeps the full scan.
_RENDERABLE_OBJECT_KINDS = frozenset({"string", "bytes", "integer", "empty"})

#: Whether the fast paths that restrict hashing to the rows that can actually
#: be truncated are enabled. Tests set this to False to check that the fast
#: and full paths produce identical frames.
_FAST_PATH_ENABLED = True


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


def _combine_digests(digests: Sequence[Optional[str]]) -> str:
    """Combines the per-value digests of one row into that row's digest.

    This mirrors Spark's ``_hash_columns``: the per-value digests are joined
    with commas, skipping nulls, and that string is hashed, and its digest
    hashed once more.

    This is the choke point every combined hash flows through, and the seam
    that four hash-collision regression tests in
    ``test.unit.utils.test_pandas_truncation`` replace with a constant. It
    must stay the single point every row's digest passes through: inlining it
    would leave those tests patching a function nothing calls.

    Returns:
        The hex-encoded SHA-256 digest for the given per-value digests.
    """
    # hashlib.sha256 is called directly rather than through _sha256: this runs
    # once per row, and the extra Python call is measurable at large sizes.
    sha256 = hashlib.sha256
    concatenated = sha256(
        ",".join(digest for digest in digests if digest is not None).encode("utf-8")
    ).hexdigest()
    return sha256(concatenated.encode("utf-8")).hexdigest()


def _combined_hash(values: Sequence[Any]) -> str:
    """Combines the hashes of ``values`` into a single hash.

    Returns:
        The hex-encoded SHA-256 digest for the given values.
    """
    return _combine_digests([_hash_value(value) for value in values])


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
        # An object column has no type of its own, so its values have to be
        # checked. An empty one carries no values and so cannot be checked,
        # unlike the Spark schema it corresponds to. When the column's
        # inferred kind proves that every value it can hold is renderable,
        # the per-value scan is skipped -- except at the positions the
        # ``skipna=True`` kind inference skipped: an NA-like value with no
        # Spark rendering, such as ``np.float16("nan")``, is invisible to
        # every kind, so the values classified as NaNs are still rendered.
        # A genuine float NaN renders as ``b"nan"``; anything else raises
        # here exactly as it does on the full scan.
        kind = _object_kind(column)
        if kind in _RENDERABLE_OBJECT_KINDS:
            _render_nan_classified_values(column.to_numpy())
            return
        if kind in ("date", "datetime"):
            # Every value is a date or a (possibly timezone-aware) datetime,
            # where two equal values always render identically or both
            # raise -- a timezone-aware datetime never equals a naive one --
            # so rendering one value per distinct value is exactly the full
            # scan, at one rendering per *distinct* date rather than per row.
            # The first failing distinct value, in order of first appearance,
            # is the first failing value of the full scan, so the error is
            # the same one. NA-like values are invisible to the
            # factorization, as they are to the kind, and are checked as
            # above.
            values = column.to_numpy()
            for value in pd.factorize(values)[1]:
                _render_value(value)
            _render_nan_classified_values(values)
            return
        for value in _column_values(column):
            _render_value(value)
        return
    raise NotImplementedError(message)


def _render_nan_classified_values(values: np.ndarray) -> None:
    """Renders every value of an object array that classifies as a NaN.

    This is the validation for the values ``infer_dtype(skipna=True)`` cannot
    see. The null-classified values need no check -- they render as ``None``
    whatever they are -- but a NaN-classified value is hashed as a value, so
    it must render: a float NaN renders as ``b"nan"``, and an NA-like value
    with no Spark rendering, such as ``np.float16("nan")`` or a stray
    ``np.datetime64("NaT")``, raises :class:`NotImplementedError`.
    """
    _, nan_mask = _null_and_nan_masks(values)
    for position in np.flatnonzero(nan_mask):
        _render_value(values[position])


def _object_kind(column: pd.Series) -> str:
    """Returns the inferred kind of an object column's values.

    Returns:
        The value of ``pandas.api.types.infer_dtype(column, skipna=True)``.
    """
    return pd.api.types.infer_dtype(column, skipna=True)


def _null_and_nan_masks(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the null and NaN masks of an object array.

    ``pandas.isna`` marks ``None``, ``pd.NA``, ``pd.NaT`` and float ``NaN``
    alike, but this module treats a float ``NaN`` in an object column as a
    value that hashes to ``b"nan"`` and sorts last, and only the first three
    as nulls. The ambiguous positions -- which are few in every realistic
    frame -- are therefore resolved one at a time with :func:`_is_null`.

    Args:
        values: The object array to inspect.

    Returns:
        A boolean mask of the null positions and a boolean mask of the NaN
        positions. The two never overlap.
    """
    missing = pd.isna(values)
    null_mask = np.zeros(len(values), dtype=bool)
    nan_mask = np.zeros(len(values), dtype=bool)
    for position in np.flatnonzero(missing):
        if _is_null(values[position]):
            null_mask[position] = True
        else:
            nan_mask[position] = True
    return null_mask, nan_mask


def _nullable_int_values(column: pd.Series) -> np.ndarray:
    """Returns a nullable integer column's values, with nulls reading as 0.

    Unsigned values above ``2**63 - 1`` would not survive a cast to int64, so
    unsigned dtypes are materialized as ``uint64``. The caller separates the
    nulls out again with the column's own null mask.

    Returns:
        An int64 or uint64 array aligned with ``column``.
    """
    target: Any = (
        np.uint64 if pd.api.types.is_unsigned_integer_dtype(column.dtype) else np.int64
    )
    return column.to_numpy(target, na_value=0)


def _microsecond_keys(column: pd.Series) -> np.ndarray:
    """Returns int64 keys grouping and ordering a datetime column like Spark.

    Two rows share a key exactly when their values agree at Spark's
    microsecond resolution, and the keys ascend in Spark's timestamp order.
    A nanosecond column is floored to microseconds, merging the
    sub-microsecond distinctions Spark cannot see; numpy's cast floors toward
    negative infinity, like ``Timestamp.floor``. A column in a coarser unit
    ('s', 'ms' or 'us', which pandas 2 allows) already carries no
    sub-microsecond precision, so its own representation is the key:
    converting it to nanoseconds, as ``to_numpy("datetime64[ns]")`` would,
    silently wraps values outside the nanosecond range, such as 9999-12-31
    in a microsecond column. ``NaT`` keeps its own sentinel value.

    Returns:
        An int64 array aligned with ``column``. Only equality and relative
        order are meaningful; the unit is the column's own.
    """
    values = column.to_numpy()
    if values.dtype == np.dtype("datetime64[ns]"):
        values = values.astype("datetime64[us]")
    return values.view("int64")


def _digest_codes(column: pd.Series) -> Optional[Tuple[np.ndarray, Sequence[Any]]]:
    """Returns a factorization of a column that never merges distinct renderings.

    The contract is one-directional: two rows sharing a code must render to
    the same bytes, but two rows that render alike may still get different
    codes. That makes over-splitting harmless and lets floating point columns
    be factorized by bit pattern, which is what keeps ``0.0`` and ``-0.0``
    apart.

    Args:
        column: The column to factorize.

    Returns:
        The per-row codes, with :data:`_NULL_DIGEST_CODE` marking nulls,
        together with one representative value per non-negative code -- at the
        precision of the column's dtype, as :func:`_column_values` yields it.
        Returns None when the column's dtype has no faithful factorization, in
        which case the caller renders every value.
    """
    dtype = column.dtype
    if isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
        float_dtype, bits_dtype = (
            (np.float32, np.int32)
            if isinstance(dtype, pd.Float32Dtype)
            else (np.float64, np.int64)
        )
        bits = column.to_numpy(float_dtype, na_value=0.0).view(bits_dtype)
        codes, uniques = pd.factorize(bits)
        codes[column.isna().to_numpy()] = _NULL_DIGEST_CODE
        return codes, uniques.view(float_dtype)
    if pd.api.types.is_integer_dtype(dtype) and not isinstance(dtype, np.dtype):
        codes, uniques = pd.factorize(_nullable_int_values(column))
        codes[column.isna().to_numpy()] = _NULL_DIGEST_CODE
        return codes, uniques
    if isinstance(dtype, pd.StringDtype):
        strings = column.to_numpy(object, na_value=None)
        codes, uniques = pd.factorize(strings)
        # pd.factorize marks the None positions with -1, which is exactly
        # _NULL_DIGEST_CODE; a string dtype can hold no NaN value.
        return codes, uniques
    if not isinstance(dtype, np.dtype):
        return None
    if dtype.kind in "iu":
        codes, uniques = pd.factorize(column.to_numpy())
        return codes, uniques
    if dtype in _SUPPORTED_FLOAT_DTYPES:
        # Factorizing the bit pattern separates 0.0 from -0.0, which render
        # differently, and splits NaN payloads, which is a harmless
        # over-split. The representatives keep the column's dtype: a float32
        # rendered with the double formatter would gain digits the float
        # never had.
        bits_dtype = np.int32 if dtype == np.dtype("float32") else np.int64
        codes, uniques = pd.factorize(column.to_numpy().view(bits_dtype))
        return codes, uniques.view(dtype)
    if pd.api.types.is_datetime64_dtype(dtype):
        # The factorization stays in the column's own unit: converting to
        # nanoseconds first would silently wrap values outside the nanosecond
        # range, which non-nanosecond columns (pandas 2) can hold.
        values = column.to_numpy()
        codes, uniques = pd.factorize(values.view("int64"))
        codes[column.isna().to_numpy()] = _NULL_DIGEST_CODE
        # Sub-microsecond precision is deliberately kept: two values
        # rendering alike may get different codes, which merely over-splits.
        return codes, [pd.Timestamp(value) for value in uniques.view(values.dtype)]
    if dtype == np.dtype(object) and _object_kind(column) in _HOMOGENEOUS_OBJECT_KINDS:
        values = column.to_numpy()
        try:
            codes, uniques = pd.factorize(values)
        except TypeError:
            # A value pd.factorize cannot hash, such as a bytearray.
            return None
        representatives = list(uniques)
        # The null positions keep pd.factorize's missing-value code, which is
        # exactly _NULL_DIGEST_CODE, so only the NaN mask is needed here.
        _, nan_mask = _null_and_nan_masks(values)
        # pd.factorize treats a float NaN in an object array as missing, but
        # here it is a value that hashes to sha256(b"nan"), unlike a null,
        # which contributes nothing; only the null positions may keep the
        # missing-value code.
        if nan_mask.any():
            codes[nan_mask] = len(representatives)
            representatives.append(float("nan"))
        return codes, representatives
    return None


class _ColumnDigests(NamedTuple):
    """The per-row digests of one column.

    Attributes:
        digests: An object array of hex digests aligned with the column,
            holding None wherever the column holds a null.
        has_null: Whether any digest is None. This is known for free from the
            digest codes, and saves the combiner a full null scan.
    """

    digests: np.ndarray
    has_null: bool


def _column_digests(column: pd.Series) -> _ColumnDigests:
    """Hashes every value of a column, hashing each distinct value once.

    Returns:
        The per-row digests, and whether any of them is None.
    """
    codes_and_values = _digest_codes(column)
    if codes_and_values is None:
        # No faithful factorization exists, so every value is rendered, as it
        # was before deduplication.
        rendered = [_hash_value(value) for value in _column_values(column)]
        return _ColumnDigests(
            np.array(rendered, dtype=object),
            any(digest is None for digest in rendered),
        )
    codes, values = codes_and_values
    # Every distinct value is rendered, so an unsupported value in an object
    # column still raises exactly as it does on the fallback path above;
    # deduplication cannot hide an error.
    digests = np.empty(len(values) + 1, dtype=object)  # slot 0 is the null slot
    digests[0] = None
    digests[1:] = [_hash_value(value) for value in values]
    return _ColumnDigests(digests[codes + 1], bool((codes == _NULL_DIGEST_CODE).any()))


def _row_digests(columns: Sequence[_ColumnDigests], n_rows: int) -> np.ndarray:
    """Combines per-column digests into one digest per row.

    Args:
        columns: One entry per hashed column, as :func:`_column_digests`
            returns them.
        n_rows: The number of rows, which is what fixes the result's length
            when there are no columns at all.

    Returns:
        An object array of hex digests, one per row.
    """
    if not columns:
        return np.full(n_rows, _combine_digests(()), dtype=object)
    combine = _combine_digests
    arrays = [column.digests for column in columns]
    if not any(column.has_null for column in columns):
        # The null-filtering comprehension below costs about 0.25 s per
        # million rows; skip it when no column holds a null.
        return np.array([combine(row) for row in zip(*arrays)], dtype=object)
    return np.array(
        [
            combine([digest for digest in row if digest is not None])
            for row in zip(*arrays)
        ],
        dtype=object,
    )


def _hash_columns(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Hashes the given columns of every row into a single value.

    The truncation functions inline these steps around their fast paths, so
    this composition survives as the reference implementation the golden
    vectors and the hash-agreement tests pin.

    Returns:
        A series of hex-encoded SHA-256 digests, aligned with ``df``.
    """
    for column in columns:
        _validate_column(df[column], column)
    hashes = _row_digests([_column_digests(df[column]) for column in columns], len(df))
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
    if pd.api.types.is_scalar(value) and pd.isna(value):
        # An NA-like value outside the branches above -- Decimal("NaN"), or a
        # raw np.datetime64("NaT") in an object column -- compares unequal to
        # itself, so keying it by value would make every occurrence a
        # partition of its own and let an oversized group of them slip past
        # drop_large_groups. Such values are keyed like the float NaNs they
        # behave as, which is also where :func:`_null_and_nan_masks` puts
        # them on the vectorized paths.
        return (_NAN_ORDER, 0)
    return (_VALUE_ORDER, value)


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


def _dense_codes(codes: Sequence[np.ndarray]) -> np.ndarray:
    """Combines several code arrays into one dense code per distinct combination.

    Args:
        codes: The code arrays to combine, at least one.

    Returns:
        An int64 array of codes in ``range(number of distinct combinations)``,
        numbered in order of first appearance.
    """
    # The host series exists only to give groupby a frame-shaped anchor; the
    # keys carry all the information.
    return (
        pd.Series(0, index=range(len(codes[0])))
        .groupby(list(codes), sort=False, dropna=False)
        .ngroup()
        .to_numpy()
    )


def _fallback_group_codes(column: pd.Series) -> np.ndarray:
    """Returns group codes by building every row's :func:`_group_key`.

    This is the exact, per-value path for the columns the vectorized cases in
    :func:`_group_codes` cannot handle faithfully.

    Returns:
        A non-negative int64 array aligned with ``column``.
    """
    keys = pd.Series(
        [_group_key(value) for value in _column_values(column)], dtype=object
    )
    return pd.factorize(keys)[0].astype(np.int64, copy=False)


def _group_codes(column: pd.Series) -> np.ndarray:
    """Returns one dense code per Spark partition key of a column.

    Two rows share a code exactly when :func:`_group_key` gives them the same
    key, so grouping by these codes forms the partitions Spark would form.
    Unlike :func:`_digest_codes`, this factorization must be exact in both
    directions: an over-split (``0.0`` versus ``-0.0``, ``bytes`` versus
    ``bytearray``) would change which rows share a group.

    Returns:
        A non-negative int64 array aligned with ``column``.
    """
    dtype = column.dtype
    if isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
        float_dtype = np.float32 if isinstance(dtype, pd.Float32Dtype) else np.float64
        floats = column.to_numpy(float_dtype, na_value=np.nan)
        # Factorizing the values, not the bit patterns, makes 0.0 and -0.0
        # one partition, as _group_key does. NaNs come out as -1 alongside
        # the nulls and the two are then separated by their masks.
        codes, uniques = pd.factorize(floats)
        codes = codes.astype(np.int64, copy=False)
        null_mask = column.isna().to_numpy()
        nan_mask = np.isnan(floats) & ~null_mask
        codes[nan_mask] = len(uniques)
        codes[null_mask] = len(uniques) + 1
        return codes
    if pd.api.types.is_integer_dtype(dtype) and not isinstance(dtype, np.dtype):
        codes, uniques = pd.factorize(_nullable_int_values(column))
        codes = codes.astype(np.int64, copy=False)
        null_mask = column.isna().to_numpy()
        if null_mask.any():
            codes[null_mask] = len(uniques)
        return codes
    if isinstance(dtype, pd.StringDtype):
        strings = column.to_numpy(object, na_value=None)
        codes, uniques = pd.factorize(strings)
        codes = codes.astype(np.int64, copy=False)
        codes[codes == -1] = len(uniques)  # the nulls are one partition
        return codes
    if not isinstance(dtype, np.dtype):
        # Extension dtypes with no vectorized path, e.g. the categorical and
        # boolean columns drop_large_groups accepts without validation.
        return _fallback_group_codes(column)
    if dtype.kind in "iu":
        return pd.factorize(column.to_numpy())[0].astype(np.int64, copy=False)
    if dtype in _SUPPORTED_FLOAT_DTYPES:
        floats = column.to_numpy()
        codes, uniques = pd.factorize(floats)
        codes = codes.astype(np.int64, copy=False)
        codes[codes == -1] = len(uniques)  # the NaNs are one partition
        return codes
    if pd.api.types.is_datetime64_dtype(dtype):
        # NaT keeps its own sentinel value and so its own partition.
        return pd.factorize(_microsecond_keys(column))[0].astype(np.int64, copy=False)
    if dtype == np.dtype(object) and _object_kind(column) in _HOMOGENEOUS_OBJECT_KINDS:
        values = column.to_numpy()
        try:
            codes, uniques = pd.factorize(values)
        except TypeError:
            return _fallback_group_codes(column)
        codes = codes.astype(np.int64, copy=False)
        null_mask, nan_mask = _null_and_nan_masks(values)
        # A pandas groupby would put NaNs and nulls in one group; Spark makes
        # them two partitions.
        codes[nan_mask] = len(uniques)
        codes[null_mask] = len(uniques) + 1
        return codes
    return _fallback_group_codes(column)


class _OrderKeys(NamedTuple):
    """Lexsort keys reproducing Spark's ascending order for one column.

    Attributes:
        classes: One of :data:`_NULL_ORDER`, :data:`_VALUE_ORDER` or
            :data:`_NAN_ORDER` per row, or None when every row holds an
            ordinary value and the class is therefore constant.
        values: A per-row key whose ascending order is Spark's, compared only
            between rows of the same class.
    """

    classes: Optional[np.ndarray]
    values: np.ndarray


def _order_classes(
    null_mask: Optional[np.ndarray], nan_mask: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Returns the per-row order class, or None when every row is a value.

    Args:
        null_mask: The mask of null rows, or None when there can be none.
        nan_mask: The mask of NaN rows, or None when there can be none.

    Returns:
        An int8 array of order classes, or None when it would be constant.
    """
    has_nulls = null_mask is not None and null_mask.any()
    has_nans = nan_mask is not None and nan_mask.any()
    if not has_nulls and not has_nans:
        return None
    length = len(null_mask if null_mask is not None else nan_mask)  # type: ignore[arg-type]
    classes = np.full(length, _VALUE_ORDER, dtype=np.int8)
    if has_nulls:
        classes[null_mask] = _NULL_ORDER
    if has_nans:
        classes[nan_mask] = _NAN_ORDER
    return classes


def _dense_ranks(values: np.ndarray) -> np.ndarray:
    """Returns each value's dense rank in the ascending order of its uniques.

    The ranks are computed over the whole array, never over a subset, so
    restricting them to any subset of rows induces the same order there.
    Missing positions (the nulls and NaNs ``pd.factorize`` marks) rank zero;
    the caller's class key is what separates them from the values.

    Returns:
        An int64 array aligned with ``values``.
    """
    codes, _ = pd.factorize(values, sort=True)
    return np.where(codes < 0, 0, codes).astype(np.int64, copy=False)


def _fallback_order_keys(column: pd.Series) -> _OrderKeys:
    """Returns order keys for a column with no vectorized ordering.

    The ranks are those of every row's full :func:`_group_key`, in
    :func:`_sorted_keys` order, so mixed-type object columns keep the exact
    deterministic order (including the type-name fallback) they had when the
    per-value path was the only path. The class is part of the key, so no
    separate class array is needed.

    Returns:
        The order keys for the column.
    """
    keys = [_group_key(value) for value in _column_values(column)]
    ranks = {key: rank for rank, key in enumerate(_sorted_keys(set(keys)))}
    return _OrderKeys(None, np.array([ranks[key] for key in keys], dtype=np.int64))


def _order_keys(column: pd.Series) -> _OrderKeys:
    """Returns the sort keys ordering a column the way Spark orders it.

    The keys are absolute -- derived from the values themselves, or from
    dense ranks over the whole column -- so restricting them to a subset of
    the rows induces the same order on that subset. That is what lets the
    fast path compute them once, before deciding which rows to hash.

    Returns:
        The class and value keys for the column.
    """
    dtype = column.dtype
    if isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
        float_dtype = np.float32 if isinstance(dtype, pd.Float32Dtype) else np.float64
        floats = column.to_numpy(float_dtype, na_value=np.nan)
        nans = np.isnan(floats)
        null_mask = column.isna().to_numpy()
        values = np.where(nans, float_dtype(0.0), floats)
        return _OrderKeys(_order_classes(null_mask, nans & ~null_mask), values)
    if pd.api.types.is_integer_dtype(dtype) and not isinstance(dtype, np.dtype):
        null_mask = column.isna().to_numpy()
        values = _nullable_int_values(column)
        return _OrderKeys(_order_classes(null_mask, None), values)
    if isinstance(dtype, pd.StringDtype):
        strings = column.to_numpy(object, na_value=None)
        null_mask = column.isna().to_numpy()
        return _OrderKeys(_order_classes(null_mask, None), _dense_ranks(strings))
    if not isinstance(dtype, np.dtype):
        return _fallback_order_keys(column)
    if dtype.kind in "iu":
        # Any monotone key works; the raw integers are one.
        return _OrderKeys(None, column.to_numpy())
    if dtype in _SUPPORTED_FLOAT_DTYPES:
        floats = column.to_numpy()
        nan_mask = np.isnan(floats)
        # -0.0 and 0.0 compare equal in the value key, and the sort is
        # stable, which is exactly the tie _group_key gives them.
        values = np.where(nan_mask, floats.dtype.type(0.0), floats)
        return _OrderKeys(_order_classes(None, nan_mask), values)
    if pd.api.types.is_datetime64_dtype(dtype):
        null_mask = column.isna().to_numpy()
        values = np.where(null_mask, np.int64(0), _microsecond_keys(column))
        return _OrderKeys(_order_classes(null_mask, None), values)
    if dtype == np.dtype(object) and _object_kind(column) in _HOMOGENEOUS_OBJECT_KINDS:
        objects = column.to_numpy()
        null_mask, nan_mask = _null_and_nan_masks(objects)
        try:
            values = _dense_ranks(objects)
        except TypeError:
            return _fallback_order_keys(column)
        return _OrderKeys(_order_classes(null_mask, nan_mask), values)
    return _fallback_order_keys(column)


def _digest_order_key(digests: np.ndarray) -> np.ndarray:
    """Returns the sort key ordering a column of hex digests.

    Every digest is 64 ASCII characters, so a fixed-width bytes array orders
    them exactly as Python orders the strings, and compares them in C rather
    than through the Python object protocol.

    Returns:
        An ``S64`` array aligned with ``digests``.
    """
    return digests.astype("S64")


def _tie_break_keys(
    order_keys: Mapping[str, _OrderKeys], columns: Sequence[str], take: np.ndarray
) -> List[np.ndarray]:
    """Returns the tie-breaking lexsort keys for ``columns``, taken at ``take``.

    numpy's lexsort takes the last key as the primary one, so the keys run
    from the last tie-breaking column upward, each column contributing its
    value key and, above it, its class key when one exists. The caller
    supplies the digest key as the primary key.

    Args:
        order_keys: The per-column order keys, computed over the full frame.
        columns: The tie-breaking columns, from highest to lowest priority.
        take: The positions of the rows being sorted in the keys' frame.

    Returns:
        The lexsort keys, in increasing order of priority.
    """
    keys: List[np.ndarray] = []
    for column in reversed(list(columns)):
        order_key = order_keys[column]
        keys.append(order_key.values[take])
        if order_key.classes is not None:
            keys.append(order_key.classes[take])
    return keys


def _hash_sort_order(
    tie_keys: Callable[[], Sequence[np.ndarray]], digest_key: np.ndarray
) -> np.ndarray:
    """Returns the permutation sorting rows by digest, then by the tie keys.

    When every digest in the frame is distinct, the digest alone is a strict
    total order and the tie-breaking keys cannot matter, so the cheaper
    single-key sort is used and ``tie_keys`` is never called -- which is what
    lets callers defer building the order keys entirely in the common
    all-distinct case. The branch is exact, not probabilistic: it is taken
    only when no two rows share a digest, which an adjacent-duplicate check
    on the sorted digests decides. Duplicate digests do occur -- a null
    contributes nothing to the combined hash, so ``(NULL, "k1")`` and
    ``("k1", NULL)`` collide -- and then every key participates.

    Args:
        tie_keys: Returns the tie-breaking lexsort keys, in increasing order
            of priority. Called only when two rows share a digest.
        digest_key: The primary key, as :func:`_digest_order_key` returns it.

    Returns:
        The stable ascending permutation.
    """
    order = np.argsort(digest_key, kind="stable")
    sorted_key = digest_key[order]
    if not (sorted_key[1:] == sorted_key[:-1]).any():
        return order
    # numpy's lexsort takes the last key as the primary one, and is stable.
    return np.lexsort([*tie_keys(), digest_key])


def _first_occurrences(codes: np.ndarray) -> np.ndarray:
    """Returns the position of each code's first occurrence, in code order.

    ``codes`` must be first-occurrence dense -- ``pd.factorize`` or
    :func:`_dense_codes` output, numbered ``0, 1, ...`` in order of first
    appearance. A position is then a first occurrence exactly when its code
    exceeds every earlier code, and first occurrences appear in code order,
    so this equals ``np.unique(codes, return_index=True)[1]`` without the
    O(n log n) sort.

    Returns:
        An int64 array with one position per distinct code.
    """
    if not len(codes):
        return np.zeros(0, dtype=np.int64)
    is_first = np.empty(len(codes), dtype=bool)
    is_first[0] = True
    is_first[1:] = codes[1:] > np.maximum.accumulate(codes)[:-1]
    return np.flatnonzero(is_first)


def _prefix_ranks(ids: np.ndarray) -> np.ndarray:
    """Returns each element's one-based rank among prior elements of its id.

    This is the cumulative count Spark's windowed ``row_number`` produces
    once the rows stand in their final order.

    Returns:
        An int64 array aligned with ``ids``.
    """
    series = pd.Series(ids)
    return (series.groupby(series, sort=False).cumcount() + 1).to_numpy()


def _group_ids(codes: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    """Returns one dense group id per row, treating no columns as one group.

    Args:
        codes: One array per grouping column, as :func:`_group_codes` returns
            them, possibly none at all.
        n_rows: The number of rows, which fixes the result's length when
            there are no grouping columns.

    Returns:
        A non-negative int64 array aligned with the frame. Every consumer is
        label-agnostic, so a single column's codes already are its ids.
    """
    if not codes:
        return np.zeros(n_rows, dtype=np.int64)
    if len(codes) == 1:
        return codes[0]
    return _dense_codes(codes)


def _survivors_in_input_order(
    working_df: pd.DataFrame,
    columns: List[str],
    selected: np.ndarray,
    kept_positions: np.ndarray,
) -> pd.DataFrame:
    """Returns the rows surviving a truncation, in input order.

    The survivors are every row the fast path left unselected, plus the
    selected rows at ``kept_positions``. Selecting with a mask returns them
    in input order; see the module docstring's "Row order" note.

    Args:
        working_df: The frame being truncated, with a default index.
        columns: The columns of the result.
        selected: The mask of rows the truncation considered.
        kept_positions: The positions of the considered rows that survived.

    Returns:
        The surviving frame, reindexed from zero.
    """
    survivors = np.zeros(len(working_df), dtype=bool)
    survivors[~selected] = True
    survivors[kept_positions] = True
    return working_df.loc[survivors, columns].reset_index(drop=True)


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
    n = len(working_df)
    # Spark accepts a repeated partitioning column; grouping by the same
    # column twice here would only be wasted work. The group codes are built
    # before the early return so that an unknown grouping column raises a
    # KeyError whatever the threshold is.
    grouping_unique = list(dict.fromkeys(grouping_columns))
    group_code = {
        column: _group_codes(working_df[column]) for column in grouping_unique
    }
    if threshold <= 0 or n == 0:
        # Spark expresses the threshold as a filter, so a non-positive
        # threshold is an empty result, not an error.
        return working_df.iloc[:0].copy()
    group_ids = _group_ids([group_code[column] for column in grouping_unique], n)
    # The fast path: a group of size m <= threshold contributes its first
    # min(m, threshold) = m rows -- all of them -- whatever the hash order
    # turns out to be, so only the rows of oversized groups need hashing.
    # The module docstring's "Fast paths" section holds the full argument.
    sizes = np.bincount(group_ids)[group_ids]
    selected = sizes > threshold if _FAST_PATH_ENABLED else np.ones(n, dtype=bool)
    if not selected.any():
        # Every row survives. Copy, so no blocks are shared with the caller.
        return working_df.copy()
    positions = np.flatnonzero(selected)
    sub = working_df.iloc[positions]
    # Identical rows must hash differently, or they would be kept or dropped
    # as a block. Spark numbers them with row_number over a window partitioned
    # by every column, which is a cumulative count over identical rows. Each
    # all-columns partition lies within one group, so restricting the count to
    # the selected rows leaves every salt unchanged (the salt-locality step of
    # the "Fast paths" argument) -- and because the partition is intrinsic to
    # the values, the non-grouping columns' codes can be computed on the
    # selected rows directly. The salt also makes deduplicating whole rows
    # before hashing pointless: (values, salt) is unique per row by
    # construction.
    if starting_columns:
        salt = (
            sub.groupby(
                [
                    group_code[column][positions]
                    if column in group_code
                    else _group_codes(sub[column])
                    for column in starting_columns
                ],
                sort=False,
                dropna=False,
            ).cumcount()
            + 1
        ).to_numpy()
    else:
        # With no columns at all, every row is in the same partition, so the
        # full-frame count at position p is p itself.
        salt = positions + 1
    digests = _row_digests(
        [_column_digests(sub[column]) for column in starting_columns]
        + [_column_digests(pd.Series(salt))],
        len(sub),
    )

    def tie_keys() -> List[np.ndarray]:
        # The order keys are computed over the full frame and then restricted
        # (the restriction step of the "Fast paths" argument). They matter
        # only when two digests collide, so they are built lazily.
        order_keys = {
            column: _order_keys(working_df[column]) for column in starting_columns
        }
        return _tie_break_keys(order_keys, starting_columns, positions)

    order = _hash_sort_order(tie_keys, _digest_order_key(digests))
    rank = _prefix_ranks(group_ids[positions][order])
    return _survivors_in_input_order(
        working_df, starting_columns, selected, positions[order[rank <= threshold]]
    )


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
    working_df = df.reset_index(drop=True)
    grouping_unique = list(dict.fromkeys(grouping_columns))
    group_ids = _group_ids(
        [_group_codes(working_df[column]) for column in grouping_unique],
        len(working_df),
    )
    sizes = np.bincount(group_ids)[group_ids]
    kept = working_df.loc[sizes <= threshold, starting_columns]
    return kept.reset_index(drop=True)


class _RefinedPairs(NamedTuple):
    """The refined (group, digest, key) classes of the fast budget test.

    ``None`` stands in for the whole tuple when a hashed column has no
    faithful factorization and the refinement is unavailable, so a wrongly
    weakened guard crashes instead of silently merging every row into one
    class.

    Attributes:
        codes: One dense code per row, as :func:`_dense_codes` returns them.
        first: The position of each class's first occurrence, in code order.
    """

    codes: np.ndarray
    first: np.ndarray


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
    hashed = [*grouping_columns, *key_columns]
    hashed_unique = list(dict.fromkeys(hashed))
    working_df = df.reset_index(drop=True)
    for column in hashed_unique:
        _validate_column(working_df[column], column)
    n = len(working_df)
    if threshold <= 0 or n == 0:
        # Spark expresses the threshold as a filter, so a non-positive
        # threshold is an empty result, not an error.
        return working_df.iloc[:0].copy()
    grouping_unique = list(dict.fromkeys(grouping_columns))
    key_unique = list(dict.fromkeys(key_columns))
    group_code = {column: _group_codes(working_df[column]) for column in hashed_unique}
    group_ids = _group_ids([group_code[column] for column in grouping_unique], n)
    # The digest codes exist only to build the refined budget test below,
    # which is only valid when EVERY hashed column contributed codes: a
    # column that fell back to per-value rendering has none, and the refined
    # identity would then be coarser than Spark's -- it would merge, for
    # instance, an object column's 1 and 1.0, which share a group key but
    # render "1" and "1.0" -- which would UNDER-count a group's keys and skip
    # a group that needed truncating. The collection therefore stops at the
    # first such column (and is skipped entirely when the fast path is off),
    # rather than factorizing columns whose codes would be discarded unused.
    digest_code: Optional[Dict[str, Tuple[np.ndarray, Sequence[Any]]]] = (
        {} if _FAST_PATH_ENABLED else None
    )
    if digest_code is not None:
        for column in hashed_unique:
            codes_and_values = _digest_codes(working_df[column])
            if codes_and_values is None:
                digest_code = None
                break
            digest_code[column] = codes_and_values
    refined: Optional[_RefinedPairs] = None
    if digest_code is not None:
        # Rows sharing a refined code necessarily share their group key,
        # their combined digest, and their key key, so counting refined
        # classes per group can only OVER-count a group's keys, which merely
        # hashes a group that needed no hashing (see the module docstring's
        # "Fast paths" section).
        refined_codes = _dense_codes(
            [group_code[column] for column in hashed_unique]
            + [codes for codes, _ in digest_code.values()]
        )
        refined = _RefinedPairs(refined_codes, _first_occurrences(refined_codes))
        pairs_per_group = np.bincount(
            group_ids[refined.first], minlength=int(group_ids.max()) + 1
        )
        selected = pairs_per_group[group_ids] > threshold
    else:
        selected = np.ones(n, dtype=bool)
    if not selected.any():
        # Every group is within its key budget. Copy, so no blocks are
        # shared with the caller.
        return working_df.copy()
    positions = np.flatnonzero(selected)
    # Rows in one refined class share every per-column digest, so the
    # combined digest can be computed once per class and fanned back out; on
    # frames with many rows per (group, key) pair this removes most of the
    # hashing. Both branches produce bit-identical digests; the cutoff is
    # purely economic. Every refined class lies within one group and selected
    # is a union of groups, so counting the selected class representatives
    # counts the classes. The class machinery costs about a third of what
    # combining costs per row, so it pays only when it removes at least a
    # third of the rows.
    if refined is not None and 3 * int(selected[refined.first].sum()) <= 2 * len(
        positions
    ):
        class_codes = pd.factorize(refined.codes[positions])[0]
        class_first = _first_occurrences(class_codes)
        representatives = working_df.iloc[positions[class_first]]
        class_digests = _row_digests(
            [_column_digests(representatives[column]) for column in hashed],
            len(class_first),
        )
        digests = class_digests[class_codes]
    else:
        sub = working_df.iloc[positions]
        digests = _row_digests(
            [_column_digests(sub[column]) for column in hashed], len(sub)
        )
    # The hash only depends on the grouping and key columns, so all rows of a
    # (group, key) pair share it. Spark ranks the pairs with dense_rank; here
    # each pair is given an id, ranked once, and the surviving ids select rows.
    # Spark's dense_rank ranks by (hash, *key_columns), so the hash is part of
    # the pair's identity: pandas considers -0.0 and 0.0 equal keys, but they
    # hash differently and Spark counts them as two keys.
    pair_ids = _dense_codes(
        [group_code[column][positions] for column in grouping_unique]
        + [pd.factorize(digests)[0]]
        + [group_code[column][positions] for column in key_unique]
    )
    # One representative row per pair, in input order (matching the stable
    # drop_duplicates this replaces), sorted by (digest, *key_columns).
    pair_first = _first_occurrences(pair_ids)
    pair_positions = positions[pair_first]

    def tie_keys() -> List[np.ndarray]:
        # The order keys are computed over the full frame and then restricted
        # (the restriction step of the "Fast paths" argument). They matter
        # only when two pair digests collide, so they are built lazily.
        order_keys = {column: _order_keys(working_df[column]) for column in key_unique}
        return _tie_break_keys(order_keys, list(key_columns), pair_positions)

    ordered_pairs = pair_first[
        _hash_sort_order(tie_keys, _digest_order_key(digests[pair_first]))
    ]
    rank = _prefix_ranks(group_ids[positions][ordered_pairs])
    # The pair ids are 0, 1, ... and so index a mask of the surviving pairs
    # directly, which selects the rows belonging to those pairs.
    surviving = np.zeros(len(pair_first), dtype=bool)
    surviving[pair_ids[ordered_pairs[rank <= threshold]]] = True
    return _survivors_in_input_order(
        working_df, starting_columns, selected, positions[surviving[pair_ids]]
    )
