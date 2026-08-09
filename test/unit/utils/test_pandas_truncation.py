"""Tests for :mod:`~tmlt.core.utils.pandas_truncation`.

These tests never build a Spark session. What they pin instead are the frozen
golden digests in :data:`HASH_VECTORS` and :data:`COMBINED_VECTORS`, which were
minted once by running the Spark implementation, plus the value renderings,
error contracts, and frame-level invariants that the pandas implementation owes
its Spark twin. The live Spark comparison lives in
``test_truncation_differential.py``; freezing the digests here is what localizes
which of the two implementations moved when the two suites disagree.

Regenerating the golden vectors:
    The digests were produced with a local Spark session whose
    ``spark.sql.session.timeZone`` was ``UTC``, by hashing a one-row dataframe
    with the Spark helpers directly::

        from pyspark.sql.types import StructField, StructType
        from tmlt.core.utils.truncation import _hash_column, _hash_columns

        schema = StructType([StructField("c", <SparkType>(), True)])
        df = spark.createDataFrame([(value,)], schema)
        hashed, column = _hash_column(df, "c")
        print(hashed.select(column).collect()[0][column])

        # ... and for COMBINED_VECTORS, over a frame with one column per value:
        hashed, column = _hash_columns(df, ["c0", "c1", ...])
        print(hashed.select(column).collect()[0][column])

    Naive datetimes must have UTC attached before being handed to
    ``createDataFrame``, so that Spark's wall clock matches the pandas one.

    Generated against Spark 3.5.9 / OpenJDK 17.0.13 (pre-JDK-19
    ``Double.toString``). Every digest in these tables is JVM-independent: no
    vector uses a value whose rendering differs between Java 17 and Java 19, and
    ``test_java_double_to_string_prefers_java_19_rendering`` pins the one value
    tried here that does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import decimal
from test.unit.utils.truncation_testing import EDGE_CASES, EdgeCase
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from tmlt.core.utils import pandas_truncation
from tmlt.core.utils.pandas_truncation import (
    _combined_hash,
    _hash_columns,
    _hash_value,
    _java_double_to_string,
    _java_float_to_string,
    _render_value,
    _validate_column,
    drop_large_groups,
    limit_keys_per_group,
    truncate_large_groups,
)
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

# Non-ASCII strings, written as escapes so that the source stays ASCII: a
# precomposed e-acute, an ASCII e followed by a combining acute accent (which
# renders identically but is a different string), three CJK characters, and an
# emoji from outside the basic multilingual plane.
_E_ACUTE = "\u00e9"
_E_COMBINING_ACUTE = "e\u0301"
_CJK = "\u65e5\u672c\u8a9e"
_EMOJI = "\U0001f642"

################################################################################
# Frozen golden vectors
################################################################################

#: One entry per branch of the value hashing, as
#: ``(id, value, digest Spark produces for it)``. Null values are covered by
#: :func:`test_hash_value_of_null_is_none` instead, because a null vector here
#: would be indistinguishable from an unset parameter.
HASH_VECTORS: Tuple[Tuple[str, Any, str], ...] = (
    (
        "int-zero",
        0,
        "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9",
    ),
    (
        "int-one",
        1,
        "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
    ),
    (
        "int-minus-one",
        -1,
        "1bad6b8cf97131fceab8543e81f7757195fbb1d36b376ee994ad1cf17699c464",
    ),
    (
        "int-int64-min",
        -9223372036854775808,
        "85386477f3af47e4a0b308ee3b3a688df16e8b2228105dd7d4dcd42a9807cb78",
    ),
    (
        "int-int64-max",
        9223372036854775807,
        "b34a1c30a715f6bf8b7243afa7fab883ce3612b7231716bdcbbdc1982e1aed29",
    ),
    (
        "double-zero",
        0.0,
        "8aed642bf5118b9d3c859bd4be35ecac75b6e873cce34e7b6f554b06f75550d7",
    ),
    (
        "double-negative-zero",
        -0.0,
        "c26617c7ccbcaa6631b45d851b8cf56e21d2ca624bdb1193afdbd4b560702cec",
    ),
    (
        "double-one",
        1.0,
        "d0ff5974b6aa52cf562bea5921840c032a860a91a3512f7fe8f768f6bbe005f6",
    ),
    (
        "double-negative-one-and-a-half",
        -1.5,
        "37c2b212b94e5372b33df924ea2a91182d90c237d0bf942c1768e794ebef2376",
    ),
    (
        "double-tenth",
        0.1,
        "14be4b45f18e0d8c67b4f719b5144eee88497e413709d11d85b096d8e2346310",
    ),
    (
        "double-one-third",
        1 / 3,
        "e965f1b975608cb0d1dad8c30d17e0fe1bdea42df938c0bdc29d75c97b45c44b",
    ),
    (
        "double-1e-3",
        0.001,
        "9fca51987c96ba92d35f303353b7065f31114501c9f2afa37463ff1fdffe8f1f",
    ),
    (
        "double-minus-1e-3",
        -0.001,
        "8135858673c4aaaa5bc7d0620a0c16b571fb2c9b9ff196a6fd3f17480d26b9cf",
    ),
    (
        "double-9e-4",
        0.0009,
        "39e9777cd3f5c71f55ac21c453b16398e44e8efff06ee2c9d010fa42c7609275",
    ),
    (
        "double-1e7",
        1e7,
        "dc87fa681eabb0acc1da786aee07bf709f5a27e3b1164dae6867ab470941bee2",
    ),
    (
        "double-just-under-1e7",
        9999999.999,
        "ffe40044db65f64f224fe0de5ba17d3032e32d752443b931131a168f38a798bb",
    ),
    (
        "double-1e16",
        1e16,
        "7f56765670cf8ee855701cc468a533b9f1b654d953408f6d59cd92f1051b6a9e",
    ),
    (
        "double-min-subnormal",
        5e-324,
        "5bc67d7d35291e376832b3b503ec50109ba560cd7158ed16396e3656373e7887",
    ),
    (
        "double-max-finite",
        1.7976931348623157e308,
        "9873f42aae7e27f0288d1454d2a82941915f069bb69cd656cdae87e83c01e2dc",
    ),
    (
        "double-nan",
        float("nan"),
        "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0",
    ),
    (
        "double-inf",
        float("inf"),
        "e99270c4fa9f6ea70486c8a763d7519b57ce1a4a9a0c6e0ca3bec74a82e38c24",
    ),
    (
        "double-minus-inf",
        float("-inf"),
        "a079ce0bee235137008a8523c38544f9b42c1d4c9dfc0dd86f5b597280ef2ad4",
    ),
    (
        "float32-one",
        np.float32(1.0),
        "d0ff5974b6aa52cf562bea5921840c032a860a91a3512f7fe8f768f6bbe005f6",
    ),
    (
        "float32-tenth",
        np.float32(0.1),
        "14be4b45f18e0d8c67b4f719b5144eee88497e413709d11d85b096d8e2346310",
    ),
    (
        "float32-one-third",
        np.float32(1 / 3),
        "9cf9797be2f5dab5b806b85333ef675f082d2b98ac61d10b147c028f9a6660a4",
    ),
    (
        "float32-1e-3",
        np.float32(0.001),
        "9fca51987c96ba92d35f303353b7065f31114501c9f2afa37463ff1fdffe8f1f",
    ),
    (
        "float32-negative-zero",
        np.float32(-0.0),
        "c26617c7ccbcaa6631b45d851b8cf56e21d2ca624bdb1193afdbd4b560702cec",
    ),
    (
        "float32-1e7",
        np.float32(1e7),
        "dc87fa681eabb0acc1da786aee07bf709f5a27e3b1164dae6867ab470941bee2",
    ),
    (
        "float32-9e-4",
        np.float32(0.0009),
        "39e9777cd3f5c71f55ac21c453b16398e44e8efff06ee2c9d010fa42c7609275",
    ),
    (
        "float32-max-finite",
        np.float32(3.4028234663852886e38),
        "d944e13b22835c054c233032c7af1d81b6839b9dfc25af65b1e1a3c5aff30fb9",
    ),
    (
        "float32-min-subnormal",
        np.float32(1.401298464324817e-45),
        "ec72b258b098a46a104c1f52c5a9dae1ce0e61080a7b2624494144d8e2fb1d4b",
    ),
    (
        "float32-nan",
        np.float32("nan"),
        "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0",
    ),
    (
        "float32-inf",
        np.float32("inf"),
        "e99270c4fa9f6ea70486c8a763d7519b57ce1a4a9a0c6e0ca3bec74a82e38c24",
    ),
    (
        "float32-minus-inf",
        np.float32("-inf"),
        "a079ce0bee235137008a8523c38544f9b42c1d4c9dfc0dd86f5b597280ef2ad4",
    ),
    (
        "string-empty",
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        "string-abc",
        "abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ),
    (
        "string-with-comma",
        "a,b",
        "1eb7c54d52831bbfe8942af0b1c56b7409523a59ed6ca99c1174fef7eb32c1b5",
    ),
    (
        "string-precomposed-e-acute",
        _E_ACUTE,
        "4a99557e4033c3539de2eb65472017cad5f9557f7a0625a09f1c3f6e2ba69c4c",
    ),
    (
        "string-combining-e-acute",
        _E_COMBINING_ACUTE,
        "bf12767b0f2a56b2190075bae8169f656e3ce8d6357d4aff184bc6c7ea48f9f6",
    ),
    (
        "string-cjk",
        _CJK,
        "77710aedc74ecfa33685e33a6c7df5cc83004da1bdcef7fb280f5c2b2e97e0a5",
    ),
    (
        "string-emoji",
        _EMOJI,
        "d06f1525f791397809f9bc98682b5c13318eca4c3123433467fd4dffda44fd14",
    ),
    (
        "binary-empty",
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        "binary-abc",
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ),
    (
        "binary-high-bytes",
        b"\xff\xfe",
        "b3d510ef04275ca8e698e5b3cbb0ece3949ef9252f0cdc839e9ee347409a2209",
    ),
    (
        "binary-with-nul",
        b"\x00\x01\x02",
        "ae4b3280e56e2faf83f414a6e3dabe9d5fbe18976544c05fed121accb85b53fc",
    ),
    (
        "date-year-one",
        datetime.date(1, 1, 1),
        "adc54d5a38b33a0cff4fb88f4ce712e4afcf0eb5cd9f72c3e4a619fea31c46bb",
    ),
    (
        "date-three-digit-year",
        datetime.date(999, 12, 31),
        "8137c0715204af0e75f18c925fc1d11e4e2bc7da08a2aa708314768c4037bc3f",
    ),
    (
        "date-epoch",
        datetime.date(1970, 1, 1),
        "85c14296d9598554eeb207f773a614a81cdefaecbf35a0d7051f27cf07f896b3",
    ),
    (
        "date-leap-day",
        datetime.date(2024, 2, 29),
        "2b65ec693644068605c58315fc62d32e4eff6b2f515de973ce63f5bc6e3dcadf",
    ),
    (
        "date-max",
        datetime.date(9999, 12, 31),
        "524be55b2827968f281708f4173aa7344da4124cb13ad591a19b1920c4f160e6",
    ),
    (
        "timestamp-no-fraction",
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        "235bd07ced47839e7a86f2ed4df21987a164aa86b5d4b903fd28786b714e27b3",
    ),
    (
        "timestamp-half-second",
        datetime.datetime(2020, 1, 1, 0, 0, 0, 500000),
        "c63220988c595d2d060b84deb102a62140cc89f190240b07bcfc6022577ed14b",
    ),
    (
        "timestamp-six-digit-fraction",
        datetime.datetime(2020, 1, 1, 0, 0, 0, 123456),
        "87f96de21827b0723c086e813fd41346d2fd2dc505336ce9b3803dd92b9066cc",
    ),
    (
        "timestamp-one-microsecond",
        datetime.datetime(2020, 1, 1, 0, 0, 0, 1),
        "c21b034e98648dc589c4f9a86098e723f3daf0bda5364c9ceefc86df401fe3a0",
    ),
    (
        "timestamp-before-epoch",
        datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
        "a08ee17e30b05e8fdf5392b3b66b96388f12f0c5d8d875c78b62be6d8780e95c",
    ),
    (
        "timestamp-dst-spring-forward",
        datetime.datetime(2026, 3, 8, 2, 30, 0),
        "f9a51abb47a4f30b9319b34dcbd633a0e8a4277deee658cddca16ec39382af74",
    ),
    (
        "timestamp-dst-fall-back",
        datetime.datetime(2026, 11, 1, 1, 30, 0),
        "f380068a645191a077d6b52c5112c106900d5558fbc80676c4194c673c04af6a",
    ),
    (
        "timestamp-year-padding",
        datetime.datetime(1, 1, 1, 0, 0, 0),
        "b8f843d66d0bc7b3fd9a58cc649d57610d4d6a947794a119d5df1d77f604554e",
    ),
)

#: One entry per subtlety of the hash combiner, as
#: ``(id, values of one row, digest Spark produces for that row)``.
COMBINED_VECTORS: Tuple[Tuple[str, Tuple[Any, ...], str], ...] = (
    (
        "single-column",
        ("abc",),
        "bbdb08dd3f8e0a2dbd9a4f45045fdf45cebee1ac6706de3353e753234b318e78",
    ),
    (
        "two-columns",
        ("a", "b"),
        "dc576a4017603c3044b9af38548b6af0141283716dc6d8d24fde595820f0cc39",
    ),
    (
        "separator-in-left-value",
        ("a,", "b"),
        "f2c78155dd0ea8a19e5a3137a8a06db4730bf8006afdaf733818440a1b1e3570",
    ),
    (
        "separator-in-right-value",
        ("a", ",b"),
        "0ae83d8859255986da2cc16e8c69ddf474af0b05eb52b3f1637eb0a9cbe56432",
    ),
    (
        "null-skipped",
        (None, "b"),
        "6d4b2c55fe6f56637a3df13181669ca6c17e83cdaca2b609132c1e8eb1a1aad6",
    ),
    (
        "null-in-second-position",
        ("b", None),
        "6d4b2c55fe6f56637a3df13181669ca6c17e83cdaca2b609132c1e8eb1a1aad6",
    ),
    (
        "all-null",
        (None, None),
        "cd372fb85148700fa88095e3492d3f9f5beb43e555e5ff26d95f5a6adc36f8e6",
    ),
    (
        "no-columns",
        (),
        "cd372fb85148700fa88095e3492d3f9f5beb43e555e5ff26d95f5a6adc36f8e6",
    ),
    (
        "row-with-salt-one",
        ("a1", "b1", 1),
        "3dbb4051e8e6a38e5b45d7f4018b4b8db3351e6afa20e106b6b505acb6235a16",
    ),
    (
        "row-with-salt-two",
        ("a1", "b1", 2),
        "2c873184eaf592d7291bb584e077490f206fc91298a87101165b2e0c23182a4f",
    ),
    (
        "mixed-types",
        (
            "s",
            7,
            -0.0,
            np.float32(0.1),
            datetime.date(2024, 2, 29),
            datetime.datetime(2020, 1, 1, 0, 0, 0, 500000),
            b"\xff",
            None,
        ),
        "f7d6f1a047f3af49d4650d56082e28b879a92d6a729748cc4034d1cadcf5a414",
    ),
)


@parametrize(
    Case(case_id)(value=value, expected=expected)
    for case_id, value, expected in HASH_VECTORS
)
def test_hash_value_matches_spark(value: Any, expected: str):
    """Every value hashes to the digest Spark's _hash_column produces for it."""
    assert _hash_value(value) == expected


@parametrize(
    Case("none")(value=None),
    Case("pandas-na")(value=pd.NA),
    Case("pandas-nat")(value=pd.NaT),
)
def test_hash_value_of_null_is_none(value: Any):
    """Null values have no hash at all, so that the combiner can skip them."""
    assert _hash_value(value) is None


def test_hash_value_of_string_and_bytes_agree():
    """Spark hashes strings and binary values as raw bytes, so both collide.

    This is a property of the Spark implementation, not an accident of this
    one: ``sha2`` is applied to the column directly for both ``StringType`` and
    ``BinaryType``.
    """
    assert _hash_value("abc") == _hash_value(b"abc")
    assert _hash_value("abc") == _hash_value(bytearray(b"abc"))


def test_hash_value_distinguishes_lookalike_values():
    """Values that are easily conflated hash differently."""
    assert _hash_value(0.0) != _hash_value(-0.0)
    assert _hash_value("") != _hash_value(None)
    assert _hash_value(_E_ACUTE) != _hash_value(_E_COMBINING_ACUTE)
    assert _hash_value(1) != _hash_value(1.0)
    assert _hash_value(np.float32(1 / 3)) != _hash_value(1 / 3)


@parametrize(
    Case(case_id)(values=values, expected=expected)
    for case_id, values, expected in COMBINED_VECTORS
)
def test_combined_hash_matches_spark(values: Sequence[Any], expected: str):
    """Every row combines to the digest Spark's _hash_columns produces for it."""
    assert _combined_hash(values) == expected


def test_combined_hash_separates_values_containing_the_separator():
    """The per-value hashing keeps ('a,', 'b') and ('a', ',b') apart.

    Naively joining the values with a comma would give both rows ``a,b``. This
    is the collision the Spark combiner is built to avoid, and the pandas one
    has to avoid it in exactly the same way.
    """
    assert _combined_hash(("a,", "b")) != _combined_hash(("a", ",b"))


def test_combined_hash_skips_nulls():
    """Nulls contribute nothing, matching Spark's concat_ws.

    A consequence worth stating explicitly: a null in one column is
    indistinguishable from a null in another, so ``(None, 'b')`` and
    ``('b', None)`` do collide. Spark behaves the same way.
    """
    assert _combined_hash((None, "b")) == _combined_hash(("b", None))
    assert _combined_hash((None, None)) == _combined_hash(())


################################################################################
# Floating point rendering
################################################################################

_DOUBLE_RENDERINGS: Tuple[Tuple[float, str], ...] = (
    (0.0, "0.0"),
    (-0.0, "-0.0"),
    (1.0, "1.0"),
    (-1.0, "-1.0"),
    (1.5, "1.5"),
    (-1.5, "-1.5"),
    (0.1, "0.1"),
    (1 / 3, "0.3333333333333333"),
    (100.0, "100.0"),
    (123.456, "123.456"),
    (0.012, "0.012"),
    (0.0012, "0.0012"),
    # The plain notation window is [1e-3, 1e7): both ends are pinned here.
    (0.001, "0.001"),
    (-0.001, "-0.001"),
    (9.999999999e-4, "9.999999999E-4"),
    (0.0009, "9.0E-4"),
    (1e-4, "1.0E-4"),
    (1e-7, "1.0E-7"),
    (1234567.0, "1234567.0"),
    (9999999.0, "9999999.0"),
    (9999999.999, "9999999.999"),
    (1e7, "1.0E7"),
    (-1e7, "-1.0E7"),
    (12345678.0, "1.2345678E7"),
    (1e16, "1.0E16"),
    (1e21, "1.0E21"),
    # repr() of this value is '5152716558868863.0', whose trailing zero is not
    # a significant digit and must not be counted as one.
    (5152716558868863.0, "5.152716558868863E15"),
    (2.0**63, "9.223372036854776E18"),
    (5e-324, "4.9E-324"),
    (1.7976931348623157e308, "1.7976931348623157E308"),
)

_FLOAT_RENDERINGS: Tuple[Tuple[float, str], ...] = (
    (0.0, "0.0"),
    (-0.0, "-0.0"),
    (1.0, "1.0"),
    (-1.5, "-1.5"),
    (100.0, "100.0"),
    # 0.1 as a float32 is 0.100000001490116..., but the shortest float32 that
    # round-trips is 0.1, and that is what Java renders.
    (0.1, "0.1"),
    (1 / 3, "0.33333334"),
    (0.001, "0.001"),
    (0.0009, "9.0E-4"),
    (1e-4, "1.0E-4"),
    (1e7, "1.0E7"),
    (12345678.0, "1.2345678E7"),
    (16777216.0, "1.6777216E7"),
    (3.4028234663852886e38, "3.4028235E38"),
    (2.802596928649634e-45, "2.8E-45"),
    (1.401298464324817e-45, "1.4E-45"),
)


@parametrize(
    Case(rendered)(value=value, expected=rendered)
    for value, rendered in _DOUBLE_RENDERINGS
)
def test_java_double_to_string(value: float, expected: str):
    """Doubles render the way Java's Double.toString renders them."""
    assert _java_double_to_string(value) == expected


@parametrize(
    Case(rendered)(value=value, expected=rendered)
    for value, rendered in _FLOAT_RENDERINGS
)
def test_java_float_to_string(value: float, expected: str):
    """float32 values render the way Java's Float.toString renders them."""
    assert _java_float_to_string(np.float32(value)) == expected


@parametrize(
    Case("double")(values=[value for value, _ in _DOUBLE_RENDERINGS], is_double=True),
    Case("float")(values=[value for value, _ in _FLOAT_RENDERINGS], is_double=False),
)
def test_rendered_floats_round_trip(values: Sequence[float], is_double: bool):
    """Every rendering parses back to the value it was rendered from.

    Java's contract is that the rendering is a decimal that rounds to the
    original value; a rendering that did not round-trip would be wrong no
    matter what digits it contained.
    """
    for value in values:
        if is_double:
            assert float(_java_double_to_string(value)) == value
        else:
            rendered = _java_float_to_string(np.float32(value))
            assert np.float32(float(rendered)) == np.float32(value)


def test_java_double_to_string_prefers_java_19_rendering():
    """A subnormal where Java 18 and Java 19 disagree follows Java 19.

    Both ``9.9E-324`` and ``1.0E-323`` parse back to this double. Java 19's
    specification picks, among the decimals of one or two digits that round to
    the value, the one closest to it -- which is ``9.9E-324``, since the value
    is 9.88...e-324. Java 18 and earlier render it ``1.0E-323``, one of the
    cases covered by the JVM caveat in the module docstring, so no golden hash
    vector uses a value like this one.
    """
    value = 1e-323
    assert _java_double_to_string(value) == "9.9E-324"
    assert float("9.9E-324") == value
    assert float("1.0E-323") == value


################################################################################
# Date, timestamp, and binary rendering
################################################################################


@parametrize(
    Case("no-fraction")(
        value=datetime.datetime(2020, 1, 1, 12, 34, 56),
        expected=b"2020-01-01 12:34:56",
    ),
    Case("half-second")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 500000),
        expected=b"2020-01-01 00:00:00.5",
    ),
    Case("tenth-of-a-second")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 100000),
        expected=b"2020-01-01 00:00:00.1",
    ),
    Case("six-digit-fraction")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 123456),
        expected=b"2020-01-01 00:00:00.123456",
    ),
    Case("one-microsecond")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 1),
        expected=b"2020-01-01 00:00:00.000001",
    ),
    Case("all-nines-fraction")(
        value=datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
        expected=b"1969-12-31 23:59:59.999999",
    ),
    Case("dst-spring-forward")(
        # 02:30 does not exist in US Eastern on this date; timestamps are
        # hashed as their own wall clock, so that must not matter.
        value=datetime.datetime(2026, 3, 8, 2, 30, 0),
        expected=b"2026-03-08 02:30:00",
    ),
    Case("dst-fall-back")(
        # 01:30 happens twice in US Eastern on this date.
        value=datetime.datetime(2026, 11, 1, 1, 30, 0),
        expected=b"2026-11-01 01:30:00",
    ),
    Case("year-padding")(
        value=datetime.datetime(1, 2, 3, 4, 5, 6),
        expected=b"0001-02-03 04:05:06",
    ),
    Case("pandas-timestamp")(
        value=pd.Timestamp("2020-01-01 00:00:00.5"),
        expected=b"2020-01-01 00:00:00.5",
    ),
)
def test_render_timestamp(value: datetime.datetime, expected: bytes):
    """Timestamps render as a wall clock with trailing fractional zeros trimmed."""
    assert _render_value(value) == expected


def test_render_timestamp_discards_nanoseconds():
    """Sub-microsecond precision is discarded rather than rendered.

    Spark's ``TimestampType`` has microsecond resolution, so a pandas timestamp
    carrying nanoseconds has to be floored to match it.
    """
    value = pd.Timestamp("2020-01-01 00:00:00.123456789")
    assert value.nanosecond == 789
    assert _render_value(value) == b"2020-01-01 00:00:00.123456"
    assert _render_value(value) == _render_value(
        datetime.datetime(2020, 1, 1, 0, 0, 0, 123456)
    )


@parametrize(
    Case("year-one")(value=datetime.date(1, 1, 1), expected=b"0001-01-01"),
    Case("three-digit-year")(value=datetime.date(999, 12, 31), expected=b"0999-12-31"),
    Case("epoch")(value=datetime.date(1970, 1, 1), expected=b"1970-01-01"),
    Case("leap-day")(value=datetime.date(2024, 2, 29), expected=b"2024-02-29"),
    Case("max")(value=datetime.date(9999, 12, 31), expected=b"9999-12-31"),
)
def test_render_date(value: datetime.date, expected: bytes):
    """Dates render as yyyy-MM-dd, with the year padded to four digits."""
    assert _render_value(value) == expected


@parametrize(
    Case("empty")(value=b"", expected=b""),
    Case("ascii")(value=b"abc", expected=b"abc"),
    Case("high-bytes")(value=b"\xff\xfe", expected=b"\xff\xfe"),
    Case("nul-bytes")(value=b"\x00\x01\x02", expected=b"\x00\x01\x02"),
    # toPandas() returns bytearrays for a Spark binary column, so they have to
    # be accepted alongside bytes.
    Case("bytearray")(value=bytearray(b"abc"), expected=b"abc"),
)
def test_render_binary(value: bytes, expected: bytes):
    """Binary values are hashed as their raw bytes."""
    assert _render_value(value) == expected


def test_render_value_rejects_timezone_aware_datetime():
    """A timezone-aware datetime is rejected, with the conversion spelled out."""
    value = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    with pytest.raises(NotImplementedError, match="tz_localize"):
        _render_value(value)


################################################################################
# Column hashing
################################################################################


def test_hash_columns_preserves_float32_precision():
    """A float32 column is rendered from its own dtype, not as a double.

    Iterating a numpy float32 series yields Python floats, which would render
    with the digits of the widened double (``0.3333333432674408``) rather than
    the shortest float32 (``0.33333334``).
    """
    df = pd.DataFrame({"c": pd.Series([1 / 3], dtype="float32")})
    assert _hash_columns(df, ["c"]).iloc[0] == _combined_hash((np.float32(1 / 3),))
    doubles = pd.DataFrame({"c": pd.Series([1 / 3], dtype="float64")})
    assert _hash_columns(df, ["c"]).iloc[0] != _hash_columns(doubles, ["c"]).iloc[0]


@parametrize(
    Case("int64")(dtype="int64", values=[1, 2], expected=[1, 2]),
    Case("nullable-int64")(dtype="Int64", values=[1, None], expected=[1, None]),
    Case("float64")(dtype="float64", values=[1.5, float("nan")]),
    Case("nullable-float64")(dtype="Float64", values=[1.5, None]),
    Case("string-dtype")(dtype="string", values=["a", None]),
    Case("object")(dtype="object", values=["a", None]),
    Case("datetime64")(
        dtype="datetime64[ns]",
        values=[datetime.datetime(2020, 1, 1, 0, 0, 0, 500000), None],
        expected=[datetime.datetime(2020, 1, 1, 0, 0, 0, 500000), None],
    ),
)
def test_hash_columns_matches_value_hashes(
    dtype: str, values: Sequence[Any], expected: Optional[Sequence[Any]]
):
    """Hashing a column agrees with hashing its values one at a time.

    The expected values are given separately for the dtypes where the stored
    value is not the Python object that was written -- a null in an ``Int64``
    column is ``pd.NA``, for instance -- and default to the input otherwise.
    """
    df = pd.DataFrame({"c": pd.Series(values, dtype=object).astype(dtype)})
    hashes = _hash_columns(df, ["c"])
    assert list(hashes) == [_combined_hash((v,)) for v in (expected or values)]


def test_hash_columns_of_no_columns_is_constant():
    """Hashing no columns at all gives every row the same digest.

    ``truncate_large_groups`` on a frame with no columns has nothing to hash,
    and the combiner has to agree with Spark's empty ``concat_ws`` there too.
    """
    df = pd.DataFrame(index=pd.RangeIndex(3))
    hashes = _hash_columns(df, [])
    assert list(hashes) == [_combined_hash(())] * 3


################################################################################
# Error contracts
################################################################################

_UNSUPPORTED_COLUMNS: Tuple[Tuple[str, pd.Series], ...] = (
    ("bool", pd.Series([True, False], dtype="bool")),
    ("nullable-boolean", pd.Series([True, None], dtype="boolean")),
    ("timezone-aware", pd.Series(pd.to_datetime(["2020-01-01"]).tz_localize("UTC"))),
    ("timedelta", pd.Series(pd.to_timedelta([1, 2], unit="s"))),
    ("category", pd.Series(pd.Categorical(["a", "b"]))),
    ("categorical-integers", pd.Series(pd.Categorical([1, 2]))),
    ("complex", pd.Series([1 + 2j], dtype="complex128")),
    ("period", pd.Series(pd.period_range("2020-01", periods=2, freq="M"))),
    ("interval", pd.Series(pd.IntervalIndex.from_breaks([0, 1, 2]))),
    ("sparse", pd.Series(pd.arrays.SparseArray([0, 1]))),
)

_UNSUPPORTED_OBJECT_VALUES: Tuple[Tuple[str, Any], ...] = (
    ("bool", True),
    ("numpy-bool", np.bool_(True)),
    ("decimal", decimal.Decimal("1.5")),
    ("list", [1, 2]),
    ("tuple", (1, 2)),
    ("dict", {"a": 1}),
    (
        "timezone-aware-datetime",
        datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    ),
)


@parametrize(Case(case_id)(column=column) for case_id, column in _UNSUPPORTED_COLUMNS)
def test_validate_column_rejects_unsupported_dtypes(column: pd.Series):
    """A column whose dtype has no Spark counterpart is rejected by name."""
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        _validate_column(column, "c")


@parametrize(Case(case_id)(column=column) for case_id, column in _UNSUPPORTED_COLUMNS)
def test_validate_column_rejects_unsupported_dtypes_when_empty(column: pd.Series):
    """An empty column is rejected too: its dtype is what is being checked."""
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        _validate_column(column.iloc[:0], "c")


def test_validate_column_names_the_offending_column():
    """The error says which column could not be hashed."""
    with pytest.raises(NotImplementedError, match="for column flag"):
        _validate_column(pd.Series([True], dtype="bool"), "flag")


def test_validate_column_suggests_a_fix_for_timezone_aware_columns():
    """The timezone-aware error explains how to convert the column."""
    column = pd.Series(pd.to_datetime(["2020-01-01"]).tz_localize("US/Eastern"))
    with pytest.raises(NotImplementedError, match="tz_convert"):
        _validate_column(column, "t")


@parametrize(
    Case(case_id)(value=value) for case_id, value in _UNSUPPORTED_OBJECT_VALUES
)
def test_validate_column_rejects_unsupported_object_values(value: Any):
    """An object column is checked value by value, since it has no dtype of its own."""
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        _validate_column(pd.Series([value], dtype=object), "c")


def test_empty_object_column_cannot_be_validated():
    """An empty object column is accepted, even though Spark would know better.

    This is a documented divergence: a Spark ``BooleanType`` column with no rows
    still raises, but an empty pandas object column carries no values and no
    type, so there is nothing to reject.
    """
    _validate_column(pd.Series([], dtype=object), "c")
    empty = pd.DataFrame({"g": pd.Series([], dtype=object)})
    assert truncate_large_groups(empty, ["g"], 1).empty


def test_truncate_large_groups_rejects_unsupported_payload_columns():
    """truncate_large_groups hashes every column, so any of them can be rejected."""
    df = pd.DataFrame({"g": ["a", "b"], "flag": [True, False]})
    with pytest.raises(NotImplementedError, match="for column flag"):
        truncate_large_groups(df, ["g"], 1)


def test_limit_keys_per_group_ignores_unsupported_payload_columns():
    """limit_keys_per_group only hashes grouping and key columns."""
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "k": ["x", "y", "x"], "flag": [True, False, True]}
    )
    actual = limit_keys_per_group(df, ["g"], ["k"], 1)
    expected = pd.DataFrame({"g": ["a", "b"], "k": ["x", "x"], "flag": [True, True]})
    assert_dataframe_equal(actual, expected)


@parametrize(
    Case("grouping-column")(grouping=["flag"], keys=["k"]),
    Case("key-column")(grouping=["g"], keys=["flag"]),
)
def test_limit_keys_per_group_rejects_unsupported_hashed_columns(
    grouping: Sequence[str], keys: Sequence[str]
):
    """A grouping or key column with an unsupported dtype is still rejected."""
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "k": ["x", "y", "x"], "flag": [True, False, True]}
    )
    with pytest.raises(NotImplementedError, match="for column flag"):
        limit_keys_per_group(df, grouping, keys, 1)


@parametrize(
    Case("bool-payload")(column="flag", values=[True, False, True], grouping=["g"]),
    Case("bool-grouping")(column="flag", values=[True, False, True], grouping=["flag"]),
    Case("timedelta-payload")(
        column="t", values=pd.to_timedelta([1, 2, 3], unit="s"), grouping=["g"]
    ),
    Case("timedelta-grouping")(
        column="t", values=pd.to_timedelta([1, 2, 3], unit="s"), grouping=["t"]
    ),
)
def test_drop_large_groups_never_rejects_a_column(
    column: str, values: Any, grouping: Sequence[str]
):
    """drop_large_groups hashes nothing, so no dtype can make it raise."""
    df = pd.DataFrame({"g": ["a", "a", "b"], column: values})
    result = drop_large_groups(df, list(grouping), 3)
    assert len(result) == 3


################################################################################
# Thresholds, mutation, and index
################################################################################

_FUNCTION_CASES = (
    Case("truncate_large_groups")(
        call=lambda df, threshold: truncate_large_groups(df, ["g"], threshold)
    ),
    Case("drop_large_groups")(
        call=lambda df, threshold: drop_large_groups(df, ["g"], threshold)
    ),
    Case("limit_keys_per_group")(
        call=lambda df, threshold: limit_keys_per_group(df, ["g"], ["k"], threshold)
    ),
)


def _sample_frame() -> pd.DataFrame:
    """Returns a small frame with a non-default index and mixed dtypes."""
    return pd.DataFrame(
        {
            "g": ["a", "a", "a", "b"],
            "k": ["x", "y", "y", "x"],
            "v": pd.Series([1, 2, 3, 4], dtype="Int64"),
        },
        index=[10, 4, 7, 2],
    )


@parametrize(Case("zero")(threshold=0), Case("negative")(threshold=-1))
@parametrize(_FUNCTION_CASES)
def test_non_positive_threshold_keeps_nothing(
    call: Callable[[pd.DataFrame, int], pd.DataFrame], threshold: int
):
    """A threshold of zero or less keeps no rows, and does not raise.

    Spark expresses the threshold as a ``filter``, which is happy with any
    integer, so a negative threshold is an empty result rather than an error.
    """
    df = _sample_frame()
    result = call(df, threshold)
    assert result.empty
    assert list(result.columns) == list(df.columns)
    assert list(result.dtypes) == list(df.dtypes)


@parametrize(_FUNCTION_CASES)
def test_input_is_not_mutated(call: Callable[[pd.DataFrame, int], pd.DataFrame]):
    """The input frame is left exactly as it was found."""
    df = _sample_frame()
    before = df.copy(deep=True)
    call(df, 1)
    pd.testing.assert_frame_equal(df, before)
    assert list(df.columns) == list(before.columns)
    assert list(df.index) == list(before.index)


@parametrize(_FUNCTION_CASES)
def test_output_has_a_fresh_range_index(
    call: Callable[[pd.DataFrame, int], pd.DataFrame],
):
    """The result is indexed from zero, whatever the input index was."""
    df = _sample_frame()
    result = call(df, 2)
    assert isinstance(result.index, pd.RangeIndex)
    assert list(result.index) == list(range(len(result)))
    assert list(result.columns) == list(df.columns)
    assert list(result.dtypes) == list(df.dtypes)


@parametrize(_FUNCTION_CASES)
def test_repeated_calls_agree(call: Callable[[pd.DataFrame, int], pd.DataFrame]):
    """Truncation is deterministic: the same input keeps the same rows."""
    df = _sample_frame()
    expected = call(df, 2)
    for _ in range(3):
        pd.testing.assert_frame_equal(call(df, 2), expected)


################################################################################
# Hash collisions
################################################################################

_COLLIDING_HASH = "0" * 64


def test_limit_keys_per_group_hash_collisions(monkeypatch: pytest.MonkeyPatch):
    """Colliding key hashes are broken by the key columns, not by luck.

    This is the pandas counterpart of the regression test for #2455. Spark
    breaks ties in ``dense_rank`` with the key columns, so two keys whose hashes
    collide are still ranked in key order rather than being given the same rank.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combined_hash", lambda values: _COLLIDING_HASH
    )
    df = pd.DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [1, 1, 2, 2, 1, 2, 3, 4]})
    assert_dataframe_equal(
        limit_keys_per_group(df, ["A"], ["B"], 1),
        pd.DataFrame({"A": [1, 1, 2], "B": [1, 1, 1]}),
    )
    assert_dataframe_equal(
        limit_keys_per_group(df, ["A"], ["B"], 2),
        pd.DataFrame({"A": [1, 1, 1, 1, 2, 2], "B": [1, 1, 2, 2, 1, 2]}),
    )


def test_truncate_large_groups_hash_collisions(monkeypatch: pytest.MonkeyPatch):
    """Colliding row hashes fall back to the whole row, nulls first.

    Spark orders the rows of a group by the hash and then by every column, with
    nulls sorting first, so a constant hash degenerates into that ordering.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combined_hash", lambda values: _COLLIDING_HASH
    )
    df = pd.DataFrame({"A": ["a", "a", "a", "b"], "B": [None, "z", "y", "x"]})
    assert_dataframe_equal(
        truncate_large_groups(df, ["A"], 1),
        pd.DataFrame({"A": ["a", "b"], "B": [None, "x"]}),
    )
    assert_dataframe_equal(
        truncate_large_groups(df, ["A"], 2),
        pd.DataFrame({"A": ["a", "a", "b"], "B": [None, "y", "x"]}),
    )


def test_truncate_large_groups_hash_collisions_with_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    """Colliding hashes leave duplicate rows to be ordered by their values.

    The per-duplicate salt only ever reaches the hash, so when the hash is
    constant it cannot separate identical rows: the tie is broken by the row
    values, and the group is filled from the smallest row upwards.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combined_hash", lambda values: _COLLIDING_HASH
    )
    df = pd.DataFrame({"A": ["a"] * 4, "B": ["y", "x", "y", "x"]})
    assert_dataframe_equal(
        truncate_large_groups(df, ["A"], 3),
        pd.DataFrame({"A": ["a", "a", "a"], "B": ["x", "x", "y"]}),
    )


################################################################################
# Grouping and ordering
################################################################################

#: An object column holding three NaNs and three nulls, with nothing else to
#: tell the rows apart. Spark partitions it into two groups of three; a pandas
#: groupby, left to itself, would make it one group of six.
_NAN_AND_NULL_FRAME = pd.DataFrame(
    {
        "g": pd.Series(["G"] * 6, dtype=object),
        "v": pd.Series([float("nan")] * 3 + [None] * 3, dtype=object),
    }
)


def _value_labels(column: pd.Series) -> Tuple[str, ...]:
    """Returns a sorted label per value, telling NaN and null apart."""
    labels = []
    for value in column:
        if value is None or value is pd.NA:
            labels.append("null")
        elif isinstance(value, float) and np.isnan(value):
            labels.append("nan")
        else:
            labels.append(repr(value))
    return tuple(sorted(labels))


@parametrize(
    Case("threshold-2-drops-both-groups")(threshold=2, expected=0),
    Case("threshold-3-keeps-both-groups")(threshold=3, expected=6),
)
def test_drop_large_groups_separates_nan_from_null(threshold: int, expected: int):
    """A NaN and a null are different groups, as they are in Spark.

    Both groups hold three rows, so a threshold of three keeps every row and a
    threshold of two keeps none. Were the two conflated into a single group of
    six, a threshold of three would drop everything.
    """
    result = drop_large_groups(_NAN_AND_NULL_FRAME, ["v"], threshold)
    assert len(result) == expected


@parametrize(
    Case("threshold-1")(threshold=1, expected=("nan",)),
    Case("threshold-2")(threshold=2, expected=("nan", "null")),
    Case("threshold-3")(threshold=3, expected=("nan", "nan", "null")),
)
def test_truncate_large_groups_salts_nan_and_null_rows_separately(
    threshold: int, expected: Tuple[str, ...]
):
    """Identical rows are numbered within their own NaN or null group.

    The salt that separates identical rows is a row number over a partition of
    every column, so the three NaN rows are numbered 1, 2, 3 and the three null
    rows 1, 2, 3 -- not 1 through 6. The expected survivors were taken from
    Spark 3.5 (see the differential suite, which re-derives them there).
    """
    result = truncate_large_groups(_NAN_AND_NULL_FRAME, ["g"], threshold)
    assert _value_labels(result["v"]) == expected


@parametrize(
    Case("threshold-1")(threshold=1, expected=["q"]),
    Case("threshold-2")(threshold=2, expected=["q", "r"]),
    Case("threshold-3")(threshold=3, expected=["p", "q", "r"]),
)
def test_ordering_puts_nulls_first_and_nans_last(
    monkeypatch: pytest.MonkeyPatch, threshold: int, expected: List[str]
):
    """Ties are broken in Spark's ascending order, not in pandas'.

    Spark's ascending order puts nulls first and NaNs last, while pandas'
    ``na_position`` puts both in the same place. A constant hash leaves the
    ordering of the value columns to decide which rows survive.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combined_hash", lambda values: _COLLIDING_HASH
    )
    df = pd.DataFrame(
        {
            "v": pd.Series([float("nan"), None, 1.0], dtype=object),
            "w": ["p", "q", "r"],
        }
    )
    result = truncate_large_groups(df, [], threshold)
    assert sorted(result["w"]) == expected


def test_bytearrays_can_be_grouped_and_hashed():
    """A binary column of bytearrays behaves like one of bytes.

    ``toPandas()`` returns bytearrays for a binary column when Arrow is
    disabled, and a bytearray is not hashable, which is what a pandas groupby
    needs its keys to be.
    """
    values = [b"", b"\x00", b"\xff\xfe"]
    as_bytes = pd.DataFrame({"g": ["a", "a", "b"], "b": values})
    as_bytearrays = pd.DataFrame(
        {"g": ["a", "a", "b"], "b": [bytearray(value) for value in values]}
    )
    for threshold in (1, 2):
        expected = truncate_large_groups(as_bytes, ["g"], threshold)
        actual = truncate_large_groups(as_bytearrays, ["g"], threshold)
        assert [bytes(value) for value in actual["b"]] == list(expected["b"])
        assert list(actual["g"]) == list(expected["g"])
    assert len(limit_keys_per_group(as_bytearrays, ["g"], ["b"], 1)) == 2
    assert len(drop_large_groups(as_bytearrays, ["b"], 1)) == 3


def test_bytes_and_bytearrays_of_the_same_content_are_one_group():
    """Spark compares binary values by content, whatever holds them."""
    df = pd.DataFrame({"b": [b"\x01", bytearray(b"\x01"), b"\x02"]})
    assert list(drop_large_groups(df, ["b"], 1)["b"]) == [b"\x02"]


@parametrize(
    Case("threshold-2-drops-the-group")(threshold=2, expected=0),
    Case("threshold-3-keeps-the-group")(threshold=3, expected=3),
)
def test_nanoseconds_do_not_split_a_group(threshold: int, expected: int):
    """Timestamps are grouped at the resolution they are hashed at.

    Spark timestamps are microseconds, so the three values below are one value
    to Spark and hash identically here. Grouping has to discard the nanoseconds
    too, or the group would be split into three that Spark never sees.
    """
    df = pd.DataFrame(
        {
            "t": pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00.000000001"),
                    pd.Timestamp("2020-01-01 00:00:00.000000002"),
                    pd.Timestamp("2020-01-01 00:00:00.000000003"),
                ],
                dtype="datetime64[ns]",
            ),
            "v": ["p", "q", "r"],
        }
    )
    assert len(set(_hash_columns(df, ["t"]))) == 1
    assert len(drop_large_groups(df, ["t"], threshold)) == expected


################################################################################
# The curated corpus, without Spark
################################################################################


@parametrize(Case(case.id)(case=case) for case in EDGE_CASES)
def test_edge_cases_are_hashable_in_pandas(case: EdgeCase):
    """Every curated edge case runs on the pandas backend alone.

    The differential suite checks that the two backends agree; this checks the
    invariants that hold of the pandas implementation by itself, on the same
    frames, without needing a Spark session.
    """
    df = case.to_pandas()
    before = df.copy(deep=True)
    for threshold in case.thresholds:
        results = [
            truncate_large_groups(df, list(case.grouping), threshold),
            drop_large_groups(df, list(case.grouping), threshold),
            limit_keys_per_group(df, list(case.grouping), list(case.keys), threshold),
        ]
        for result in results:
            assert isinstance(result.index, pd.RangeIndex)
            assert list(result.columns) == list(df.columns)
            assert list(result.dtypes) == list(df.dtypes)
            assert len(result) <= len(df)
    pd.testing.assert_frame_equal(df, before)
