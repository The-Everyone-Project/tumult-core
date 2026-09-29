"""Unit tests for :mod:`~tmlt.core.transformations.dictionary`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present

import re
import textwrap
import unittest
from typing import Any, Dict, List, Tuple, Union
from unittest.case import TestCase
from unittest.mock import call

import numpy as np
from parameterized import parameterized

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.exceptions import DomainKeyError, UnsupportedDomainError
from tmlt.core.measurements.chaining import ChainTM
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountant,
    SequentialComposition,
)
from tmlt.core.measurements.noise_mechanisms import AddLaplaceNoise
from tmlt.core.measures import PureDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    AddRemoveIDs,
    DictMetric,
    IfGroupedBy,
    Metric,
    SymmetricDifference,
)
from tmlt.core.transformations.dictionary import (
    AugmentDictTransformation,
    CreateDictFromValue,
    GetValue,
    Subset,
    create_apply_dict_of_transformations,
    create_copy_and_transform_value,
    create_rename,
    create_transform_all_values,
    create_transform_value,
)
from tmlt.core.transformations.identity import Identity
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import (
    assert_property_immutability,
    create_mock_transformation,
    get_all_props,
)

_NUMPY_DICT_DOMAIN = DictDomain(
    {"A": NumpyIntegerDomain(), "B": NumpyIntegerDomain(), "C": NumpyIntegerDomain()}
)
_NUMPY_DICT_METRIC = DictMetric(
    {"A": AbsoluteDifference(), "B": AbsoluteDifference(), "C": AbsoluteDifference()}
)
_BAD_STRUCTURE_D_INS: List[Any] = [
    {"A": 1, "B": 2},  # missing key
    {"A": 1, "C": 3},  # missing key
    {"A": 1, "B": 2, "C": 3, "D": 4},  # extra key
    {"B": 2, "C": 3, "D": 4},  # accessed key missing
    (1, 2, 3),  # not a dict
]


class TestAugmentDictTransformation(TestCase):
    """Tests for class AugmentDictTransformation.

    Tests :class:`~tmlt.core.transformations.dictionary.AugmentDictTransformation`.
    """

    def setUp(self):
        """Test setup."""
        self.input_domain = DictDomain(
            {"A": NumpyIntegerDomain(), "B": NumpyIntegerDomain()}
        )
        self.input_metric = DictMetric(
            {"A": AbsoluteDifference(), "B": AbsoluteDifference()}
        )
        self.output_domain = DictDomain(
            {
                "A": NumpyIntegerDomain(),
                "B": NumpyIntegerDomain(),
                "K": NumpyIntegerDomain(),
            }
        )
        self.output_metric = DictMetric(
            {
                "A": AbsoluteDifference(),
                "B": AbsoluteDifference(),
                "K": AbsoluteDifference(),
            }
        )
        self.transformation_output_domain = DictDomain({"K": NumpyIntegerDomain()})
        self.transformation_output_metric = DictMetric({"K": AbsoluteDifference()})
        self.get_mock_transformation = lambda **kwargs: create_mock_transformation(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            output_domain=self.transformation_output_domain,
            output_metric=self.transformation_output_metric,
            **kwargs,
        )

    @parameterized.expand(get_all_props(AugmentDictTransformation))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = AugmentDictTransformation(
            transformation=self.get_mock_transformation()
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Tests that AugmentDictTransformation has expected properties."""
        inner_transformation = self.get_mock_transformation()
        transformation = AugmentDictTransformation(transformation=inner_transformation)
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, self.input_metric)
        self.assertEqual(transformation.output_domain, self.output_domain)
        self.assertEqual(transformation.output_metric, self.output_metric)
        self.assertEqual(transformation.inner_transformation, inner_transformation)

    @parameterized.expand(
        [
            (
                stability_function_implemented,
                stability_function_return_value,
                stability_relation_return_value,
            )
            for stability_function_implemented in [True, False]
            for stability_function_return_value, stability_relation_return_value in [
                ({"K": 2}, True),
                ({"K": 3}, False),
            ]
        ]
    )
    def test_stability_function_and_relation(
        self,
        stability_function_implemented: bool,
        stability_function_return_value: Dict[Any, ExactNumberInput],
        stability_relation_return_value: bool,
    ):
        """Tests that the stability function and relation work correctly."""
        inner_transformation = self.get_mock_transformation(
            stability_function_implemented=stability_function_implemented,
            stability_function_return_value=stability_function_return_value,
            stability_relation_return_value=stability_relation_return_value,
        )
        transformation = AugmentDictTransformation(transformation=inner_transformation)
        if not stability_function_implemented:
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                transformation.stability_function({"A": 1, "B": 2})
        else:
            self.assertEqual(
                transformation.stability_function({"A": 1, "B": 2}),
                {"A": 1, "B": 2, **stability_function_return_value},
            )
        self.assertEqual(
            transformation.stability_relation(
                {"A": 1, "B": 2}, {"A": 1, "B": 2, "K": 2}
            ),
            stability_relation_return_value,
        )

    def test_format(self):
        """AugmentDictTransformation formats with its inner transformation."""
        inner = GetValue(
            self.input_domain, self.input_metric, key="A"
        ) | CreateDictFromValue(NumpyIntegerDomain(), AbsoluteDifference(), key="K")
        transformation = AugmentDictTransformation(inner)
        assert transformation.format() == textwrap.dedent("""\
            AugmentDictTransformation
              ┌ GetValue key='A'
              └ CreateDictFromValue key='K'""")

    def test_correctness(self):
        """Tests that AugmentDictTransformation works correctly."""
        inner_transformation = self.get_mock_transformation(
            return_value={"K": np.int64(2)}
        )
        transformation = AugmentDictTransformation(transformation=inner_transformation)
        input_dict = {"A": np.int64(20), "B": np.int64(123)}

        actual = transformation(input_dict)
        self.assertEqual(
            inner_transformation.mock_calls,
            [call({"A": np.int64(20), "B": np.int64(123)})],
        )
        self.assertEqual(actual, {**input_dict, "K": np.int64(2)})

    @parameterized.expand(
        [
            (
                "Invalid transformation input domain: Must be a DictDomain",
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                DictDomain({"K": NumpyIntegerDomain()}),
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
            ),
            (
                "Invalid transformation output domain: Must be a DictDomain",
                DictDomain({"A": NumpyIntegerDomain(), "B": NumpyIntegerDomain()}),
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                NumpyIntegerDomain(),
                AbsoluteDifference(),
            ),
            (
                "Invalid transformation output domain. Contains overlapping keys",
                DictDomain({"A": NumpyIntegerDomain(), "B": NumpyIntegerDomain()}),
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                DictDomain({"B": NumpyIntegerDomain()}),
                DictMetric({"B": AbsoluteDifference()}),
            ),
        ]
    )
    def test_invalid_arguments_raises_error(
        self,
        error_regex: str,
        input_domain: Domain,
        input_metric: Metric,
        output_domain: Domain,
        output_metric: Metric,
    ):
        """Tests that AugmentDictTransformation raises errors appropriately."""
        inner_transformation = create_mock_transformation(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        with self.assertRaisesRegex((ValueError, UnsupportedDomainError), error_regex):
            AugmentDictTransformation(transformation=inner_transformation)


class TestGetValue(TestCase):
    """Tests for :class:`~tmlt.core.transformations.dictionary.GetValue`."""

    @parameterized.expand(get_all_props(GetValue))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = GetValue(
            input_domain=DictDomain(
                {
                    "A": NumpyIntegerDomain(),
                    ("B", "B"): NumpyIntegerDomain(),
                    "C": NumpyFloatDomain(),
                }
            ),
            input_metric=DictMetric(
                {
                    "A": AbsoluteDifference(),
                    ("B", "B"): AbsoluteDifference(),
                    "C": AbsoluteDifference(),
                }
            ),
            key=("B", "B"),
        )
        assert_property_immutability(transformation, prop_name)

    @parameterized.expand(
        [
            (
                DictMetric(
                    {
                        "key1": IfGroupedBy(["A"], SymmetricDifference()),
                        "key2": IfGroupedBy(["A"], SymmetricDifference()),
                    }
                ),
                IfGroupedBy(["A"], SymmetricDifference()),
            ),
            (
                AddRemoveIDs({"key1": "A", "key2": "A"}),
                IfGroupedBy(["A"], SymmetricDifference()),
            ),
        ]
    )
    def test_properties(
        self, input_metric: Union[DictMetric, AddRemoveIDs], output_metric: Metric
    ):
        """Tests that GetValue has expected properties."""
        input_domain = DictDomain(
            {
                "key1": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkFloatColumnDescriptor(
                            allow_nan=True, allow_inf=True, allow_null=True
                        ),
                        "C": SparkStringColumnDescriptor(),
                    }
                ),
                "key2": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "D": SparkIntegerColumnDescriptor(),
                    }
                ),
            }
        )
        transformation = GetValue(
            input_domain=input_domain, input_metric=input_metric, key="key1"
        )
        self.assertEqual(transformation.input_domain, input_domain)
        self.assertEqual(transformation.input_metric, input_metric)
        self.assertEqual(transformation.output_domain, input_domain["key1"])
        self.assertEqual(transformation.output_metric, output_metric)
        self.assertEqual(transformation.key, "key1")

    def test_format(self):
        """GetValue formats as its class name with its key."""
        transformation = GetValue(
            input_domain=DictDomain(
                {"A": NumpyIntegerDomain(), "B": NumpyIntegerDomain()}
            ),
            input_metric=DictMetric(
                {"A": AbsoluteDifference(), "B": AbsoluteDifference()}
            ),
            key="A",
        )
        assert transformation.format() == "GetValue key='A'"

    @parameterized.expand([("A", np.int_(20)), (("B", "B"), np.int_(123))])
    def test_correctness(self, key: Any, expected: Any):
        """Tests that GetValue correctly applies transformation."""
        transformation = GetValue(
            input_domain=DictDomain(
                {"A": NumpyIntegerDomain(), ("B", "B"): NumpyIntegerDomain()}
            ),
            input_metric=DictMetric(
                {"A": AbsoluteDifference(), ("B", "B"): AbsoluteDifference()}
            ),
            key=key,
        )
        actual = transformation({"A": np.int_(20), ("B", "B"): np.int_(123)})
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            (
                (
                    "Input metric DictMetric(key_to_metric={'B': AbsoluteDifference()})"
                    " and input domain DictDomain(key_to_domain={'A':"
                    " NumpyIntegerDomain(size=64)}) are not compatible."
                ),
                DictDomain({"A": NumpyIntegerDomain()}),
                DictMetric({"B": AbsoluteDifference()}),
                "A",
            ),
            (
                "'B' is not one of the input domain's keys",
                DictDomain({"A": NumpyIntegerDomain()}),
                DictMetric({"A": AbsoluteDifference()}),
                "B",
            ),
        ]
    )
    def test_invalid_arguments_raises_error(
        self,
        error_regex: str,
        input_domain: DictDomain,
        input_metric: DictMetric,
        key: str,
    ):
        """Tests that GetValue raises errors appropriately."""
        with self.assertRaisesRegex(
            (ValueError, DomainKeyError), re.escape(error_regex)
        ):
            GetValue(input_domain=input_domain, input_metric=input_metric, key=key)

    @parameterized.expand(
        [
            (
                DictMetric(
                    {
                        "key1": IfGroupedBy(["A"], SymmetricDifference()),
                        "key2": IfGroupedBy(["A"], SymmetricDifference()),
                    }
                ),
                {"key1": 2, "key2": 3},
                2,
            ),
            (AddRemoveIDs({"key1": "A", "key2": "A"}), 2, 2),
        ]
    )
    def test_stability_function_and_relation(
        self,
        input_metric: Union[DictMetric, AddRemoveIDs],
        d_in: Any,
        d_out: ExactNumberInput,
    ):
        """Tests that GetValue's stability function and relation are correct."""
        input_domain = DictDomain(
            {
                "key1": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkFloatColumnDescriptor(
                            allow_nan=True, allow_inf=True, allow_null=True
                        ),
                        "C": SparkStringColumnDescriptor(),
                    }
                ),
                "key2": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "D": SparkIntegerColumnDescriptor(),
                    }
                ),
            }
        )
        transformation = GetValue(
            input_domain=input_domain, input_metric=input_metric, key="key1"
        )
        self.assertEqual(transformation.stability_function(d_in=d_in), d_out)
        self.assertTrue(transformation.stability_relation(d_in=d_in, d_out=d_out))

    def test_stability_function_validates_only_accessed_key(self):
        """GetValue validates the key set of d_in, but only the accessed value.

        This is an intentional behavior change: invalid distances for keys other
        than :attr:`~.GetValue.key` are no longer rejected by GetValue, because they
        cannot affect its output d_out.
        """
        transformation = GetValue(
            input_domain=_NUMPY_DICT_DOMAIN, input_metric=_NUMPY_DICT_METRIC, key="A"
        )
        self.assertEqual(transformation.stability_function({"A": 1, "B": 2, "C": 3}), 1)
        # Invalid sibling distances are ignored.
        self.assertEqual(
            transformation.stability_function({"A": 1, "B": -1, "C": "x"}), 1
        )
        self.assertTrue(
            transformation.stability_relation({"A": 1, "B": -1, "C": "x"}, 1)
        )
        self.assertFalse(
            transformation.stability_relation({"A": 2, "B": -1, "C": "x"}, 1)
        )
        # An invalid distance for the accessed key is still rejected.
        for bad_value in (-1, "x", 1.5):
            with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
                transformation.stability_function({"A": bad_value, "B": 2, "C": 3})
            with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
                transformation.stability_relation({"A": bad_value, "B": 2, "C": 3}, 1)
        # An invalid d_in structure is still rejected.
        for bad_d_in in _BAD_STRUCTURE_D_INS:
            with self.assertRaisesRegex(ValueError, "DictMetric value"):
                transformation.stability_function(bad_d_in)


class TestSubset(TestCase):
    """Tests for :class:`~tmlt.core.transformations.dictionary.Subset`."""

    def setUp(self):
        """Setup."""
        self.input_domain = DictDomain(
            {
                "key1": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkFloatColumnDescriptor(
                            allow_nan=True, allow_inf=True, allow_null=True
                        ),
                        "C": SparkStringColumnDescriptor(),
                    }
                ),
                "key2": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "D": SparkIntegerColumnDescriptor(),
                    }
                ),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        keys = ["key1", "key2"]
        transformation = Subset(
            input_domain=self.input_domain,
            input_metric=AddRemoveIDs({"key1": "A", "key2": "A"}),
            keys=keys,
        )
        keys[1] = "key3"
        self.assertListEqual(transformation.keys, ["key1", "key2"])

    @parameterized.expand(get_all_props(Subset))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = Subset(
            input_domain=self.input_domain,
            input_metric=DictMetric(
                {"key1": SymmetricDifference(), "key2": SymmetricDifference()}
            ),
            keys=["key1"],
        )
        assert_property_immutability(transformation, prop_name)

    @parameterized.expand(
        [
            (
                DictMetric(
                    {"key1": SymmetricDifference(), "key2": SymmetricDifference()}
                ),
                DictMetric({"key2": SymmetricDifference()}),
            ),
            (AddRemoveIDs({"key1": "A", "key2": "A"}), AddRemoveIDs({"key2": "A"})),
        ]
    )
    def test_properties(
        self,
        input_metric: Union[DictMetric, AddRemoveIDs],
        output_metric: Union[DictMetric, AddRemoveIDs],
    ):
        """Tests that Subset has expected properties."""
        transformation = Subset(
            input_domain=self.input_domain, input_metric=input_metric, keys=["key2"]
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, input_metric)
        self.assertEqual(
            transformation.output_domain,
            DictDomain({"key2": self.input_domain["key2"]}),
        )
        self.assertEqual(transformation.output_metric, output_metric)
        self.assertEqual(transformation.keys, ["key2"])

    def test_format(self):
        """Subset formats as its class name with its keys."""
        transformation = Subset(
            input_domain=self.input_domain,
            input_metric=DictMetric(
                {"key1": SymmetricDifference(), "key2": SymmetricDifference()}
            ),
            keys=["key1"],
        )
        assert transformation.format() == "Subset keys=['key1']"

    @parameterized.expand(
        [
            (
                ["A", ("B", "B"), "C"],
                {"A": np.int_(20), ("B", "B"): "XYZ", "C": np.float64(10.11)},
            ),
            (["A"], {"A": np.int_(20)}),
            ([("B", "B"), "C"], {("B", "B"): "XYZ", "C": np.float64(10.11)}),
        ]
    )
    def test_correctness(
        self,
        keys: List[Union[str, Tuple[str, str]]],
        expected: Dict[Union[str, Tuple[str, str]], Any],
    ):
        """Tests that Subset correctly applies transformation."""
        input_domain = DictDomain(
            {
                "A": NumpyIntegerDomain(),
                ("B", "B"): NumpyIntegerDomain(),
                "C": NumpyFloatDomain(),
            }
        )
        input_metric = DictMetric(
            {
                "A": AbsoluteDifference(),
                ("B", "B"): AbsoluteDifference(),
                "C": AbsoluteDifference(),
            }
        )
        transformation = Subset(
            input_domain=input_domain, input_metric=input_metric, keys=keys
        )
        actual = transformation(
            {"A": np.int_(20), ("B", "B"): "XYZ", "C": np.float64(10.11)}
        )
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            (
                "Input metric invalid for input domain",
                DictDomain({"A": NumpyIntegerDomain()}),
                DictMetric({("B", "B"): AbsoluteDifference()}),
                ["A"],
            ),
            (
                "Invalid keys",
                DictDomain({"A": NumpyIntegerDomain()}),
                DictMetric({"A": AbsoluteDifference()}),
                ["B"],
            ),
            (
                "No keys provided",
                DictDomain({"A": NumpyIntegerDomain()}),
                DictMetric({"A": AbsoluteDifference()}),
                [],
            ),
            (
                (
                    "Input metric AddRemoveIDs(df_to_id_column={'A': 'B'}) and input"
                    " domain DictDomain(key_to_domain={'A':"
                    " NumpyIntegerDomain(size=64)}) are not compatible."
                ),
                DictDomain({"A": NumpyIntegerDomain()}),
                AddRemoveIDs({"A": "B"}),
                ["A"],
            ),
        ]
    )
    def test_invalid_arguments_raises_error(
        self,
        error_message: str,
        input_domain: DictDomain,
        input_metric: DictMetric,
        keys: List[str],
    ):
        """Tests that Subset raises errors appropriately."""
        with self.assertRaisesRegex(
            (ValueError, DomainKeyError), re.escape(error_message)
        ):
            Subset(input_domain=input_domain, input_metric=input_metric, keys=keys)

    @parameterized.expand(
        [
            (
                DictMetric(
                    {"key1": SymmetricDifference(), "key2": SymmetricDifference()}
                ),
                {"key1": 3, "key2": 10},
                {"key1": 3},
            ),
            (AddRemoveIDs({"key1": "A", "key2": "A"}), 4, 4),
        ]
    )
    def test_stability_function_and_relation(
        self, input_metric: Union[DictMetric, AddRemoveIDs], d_in: Any, d_out: Any
    ):
        """Tests that Subset's stability function and relation are correct."""
        transformation = Subset(
            input_domain=self.input_domain, input_metric=input_metric, keys=["key1"]
        )
        self.assertEqual(transformation.stability_function(d_in), d_out)

    def test_stability_function_validates_only_selected_keys(self):
        """Subset validates the key set of d_in, but only the selected values.

        This is an intentional behavior change: invalid distances for keys that are
        not in :attr:`~.Subset.keys` are no longer rejected by Subset, because they
        cannot affect its output d_out.
        """
        transformation = Subset(
            input_domain=_NUMPY_DICT_DOMAIN,
            input_metric=_NUMPY_DICT_METRIC,
            keys=["A", "B"],
        )
        self.assertEqual(
            transformation.stability_function({"A": 1, "B": 2, "C": 3}),
            {"A": 1, "B": 2},
        )
        # Invalid distances for keys that are not selected are ignored, and are
        # not part of the output.
        self.assertEqual(
            transformation.stability_function({"A": 1, "B": 2, "C": -1}),
            {"A": 1, "B": 2},
        )
        self.assertTrue(
            transformation.stability_relation(
                {"A": 1, "B": 2, "C": -1}, {"A": 1, "B": 2}
            )
        )
        # An invalid distance for any selected key is still rejected.
        for bad_d_in in (
            {"A": -1, "B": 2, "C": 3},
            {"A": 1, "B": -1, "C": 3},
            {"A": 1, "B": "x", "C": 3},
        ):
            with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
                transformation.stability_function(bad_d_in)
        # An invalid d_in structure is still rejected.
        for bad_d_in in _BAD_STRUCTURE_D_INS:
            with self.assertRaisesRegex(ValueError, "DictMetric value"):
                transformation.stability_function(bad_d_in)


class TestPartialDictValidationInContext(TestCase):
    """Invalid sibling distances ignored by GetValue are rejected elsewhere.

    :class:`~.GetValue` and :class:`~.Subset` only validate the distances they
    return. These tests check that the components which use the whole distance
    dictionary still validate all of it.
    """

    def setUp(self):
        """Setup."""
        self.get_value = GetValue(
            input_domain=_NUMPY_DICT_DOMAIN, input_metric=_NUMPY_DICT_METRIC, key="A"
        )
        self.measurement = ChainTM(
            self.get_value, AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1)
        )
        self.invalid_sibling = {"A": 1, "B": -1, "C": 3}

    def test_chain_privacy_relation_validates_whole_dict(self):
        """ChainTM's privacy relation validates every distance in d_in."""
        self.assertTrue(self.measurement.privacy_relation({"A": 1, "B": 2, "C": 3}, 1))
        with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
            self.measurement.privacy_relation(self.invalid_sibling, 1)

    def test_chain_privacy_function_uses_only_accessed_key(self):
        """ChainTM's privacy function only depends on the accessed distance."""
        self.assertEqual(self.measurement.privacy_function(self.invalid_sibling), 1)
        with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
            self.measurement.privacy_function({"A": -1, "B": 2, "C": 3})

    def test_dict_metric_validate_and_compare_reject_invalid_sibling(self):
        """DictMetric.validate and DictMetric.compare validate every distance."""
        with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
            _NUMPY_DICT_METRIC.validate(self.invalid_sibling)
        with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
            _NUMPY_DICT_METRIC.compare(self.invalid_sibling, {"A": 1, "B": 2, "C": 3})

    def test_augment_dict_rejects_invalid_sibling(self):
        """Transformations that return the whole dictionary validate all of it."""
        copy_and_transform = create_copy_and_transform_value(
            input_domain=_NUMPY_DICT_DOMAIN,
            input_metric=_NUMPY_DICT_METRIC,
            key="A",
            new_key="D",
            transformation=Identity(
                metric=AbsoluteDifference(), domain=NumpyIntegerDomain()
            ),
            hint=lambda d_in, _: d_in,
        )
        self.assertEqual(
            copy_and_transform.stability_function({"A": 1, "B": 2, "C": 3}),
            {"A": 1, "B": 2, "C": 3, "D": 1},
        )
        with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
            copy_and_transform.stability_function(self.invalid_sibling)

    def test_interactive_measurement_rejects_invalid_sibling(self):
        """The d_in of an interactive measurement is validated in full."""
        with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
            SequentialComposition(
                input_domain=_NUMPY_DICT_DOMAIN,
                input_metric=_NUMPY_DICT_METRIC,
                output_measure=PureDP(),
                d_in=self.invalid_sibling,
                privacy_budget=1,
            )

    def test_privacy_accountant_transform_in_place(self):
        """transform_in_place with GetValue sets d_in to the accessed distance."""
        accountant = PrivacyAccountant.launch(
            measurement=SequentialComposition(
                input_domain=_NUMPY_DICT_DOMAIN,
                input_metric=_NUMPY_DICT_METRIC,
                output_measure=PureDP(),
                d_in={"A": 1, "B": 2, "C": 3},
                privacy_budget=1,
            ),
            data={"A": np.int64(10), "B": np.int64(20), "C": np.int64(30)},
        )
        accountant.transform_in_place(
            GetValue(
                input_domain=_NUMPY_DICT_DOMAIN,
                input_metric=_NUMPY_DICT_METRIC,
                key="B",
            )
        )
        self.assertEqual(accountant.input_metric, AbsoluteDifference())
        self.assertEqual(accountant.d_in, 2)

    def test_nested_dict_metric_validates_accessed_value_in_full(self):
        """A DictMetric distance returned by GetValue is validated in full."""
        inner_metric = DictMetric(
            {"a": AbsoluteDifference(), "b": AbsoluteDifference()}
        )
        transformation = GetValue(
            input_domain=DictDomain(
                {
                    "X": DictDomain(
                        {"a": NumpyIntegerDomain(), "b": NumpyIntegerDomain()}
                    ),
                    "Y": NumpyIntegerDomain(),
                }
            ),
            input_metric=DictMetric({"X": inner_metric, "Y": AbsoluteDifference()}),
            key="X",
        )
        self.assertEqual(
            transformation.stability_function({"X": {"a": 1, "b": 2}, "Y": -1}),
            {"a": 1, "b": 2},
        )
        for bad_inner in ({"a": 1, "b": -1}, {"a": 1}, {"a": 1, "b": 2, "c": 3}):
            with self.assertRaisesRegex(ValueError, "Invalid (value for )?DictMetric"):
                transformation.stability_function({"X": bad_inner, "Y": 1})

    def test_dict_metric_subclass_overriding_validate(self):
        """GetValue runs the full validate of a DictMetric subclass that adds checks.

        See :meth:`~.DictMetric.validate_subset`.
        """

        class _BoundedSumDictMetric(DictMetric):
            """DictMetric whose distances must also sum to at most 10."""

            def validate(self, value: Dict[Any, Any]) -> None:
                super().validate(value)
                if sum(value.values()) > 10:
                    raise ValueError("Distances must sum to at most 10.")

        transformation = GetValue(
            input_domain=_NUMPY_DICT_DOMAIN,
            input_metric=_BoundedSumDictMetric(_NUMPY_DICT_METRIC.key_to_metric),
            key="A",
        )
        self.assertEqual(transformation.stability_function({"A": 1, "B": 2, "C": 3}), 1)
        with self.assertRaisesRegex(ValueError, "sum to at most 10"):
            transformation.stability_function({"A": 1, "B": 20, "C": 3})
        with self.assertRaisesRegex(ValueError, "Invalid value for DictMetric"):
            transformation.stability_function(self.invalid_sibling)


class TestCreateDictFromValue(TestCase):
    """Tests for class CreateDictFromValue.

    Tests :class:`~tmlt.core.transformations.dictionary.CreateDictFromValue`.
    """

    @parameterized.expand(get_all_props(CreateDictFromValue))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = CreateDictFromValue(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            key="X",
        )
        assert_property_immutability(transformation, prop_name)

    @parameterized.expand(
        [
            (
                False,
                DictMetric({("X", "Y"): IfGroupedBy(["A"], SymmetricDifference())}),
            ),
            (True, AddRemoveIDs({("X", "Y"): "A"})),
        ]
    )
    def test_properties(
        self, use_add_remove_ids: bool, output_metric: Union[DictMetric, AddRemoveIDs]
    ):
        """Tests that CreateDictFromValue has expected properties."""
        input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(),
                "B": SparkFloatColumnDescriptor(
                    allow_nan=True, allow_inf=True, allow_null=True
                ),
                "C": SparkStringColumnDescriptor(),
            }
        )
        transformation = CreateDictFromValue(
            input_domain=input_domain,
            input_metric=IfGroupedBy(["A"], SymmetricDifference()),
            key=("X", "Y"),
            use_add_remove_ids=use_add_remove_ids,
        )
        self.assertEqual(transformation.input_domain, input_domain)
        self.assertEqual(
            transformation.input_metric, IfGroupedBy(["A"], SymmetricDifference())
        )
        self.assertEqual(
            transformation.output_domain, DictDomain({("X", "Y"): input_domain})
        )
        self.assertEqual(transformation.output_metric, output_metric)
        self.assertEqual(transformation.key, ("X", "Y"))

    def test_format(self):
        """CreateDictFromValue formats as its class name with its key."""
        transformation = CreateDictFromValue(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            key="X",
        )
        assert transformation.format() == "CreateDictFromValue key='X'"

    def test_correctness(self):
        """Tests that CreateDictFromValue correctly applies transformation."""
        transformation = CreateDictFromValue(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            key="X",
        )
        actual = transformation(np.int64(20))
        expected = {"X": np.int64(20)}
        self.assertEqual(actual, expected)

    @parameterized.expand([(False, 3, {"X": 3}), (True, 4, 4)])
    def test_stability_function_and_relation(
        self, use_add_remove_ids: bool, d_in: Any, d_out: Any
    ):
        """Tests that the stability function and relation are correct."""
        input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(),
                "B": SparkFloatColumnDescriptor(
                    allow_nan=True, allow_inf=True, allow_null=True
                ),
                "C": SparkStringColumnDescriptor(),
            }
        )
        transformation = CreateDictFromValue(
            input_domain=input_domain,
            input_metric=IfGroupedBy(["A"], SymmetricDifference()),
            key="X",
            use_add_remove_ids=use_add_remove_ids,
        )
        self.assertEqual(transformation.stability_function(d_in), d_out)
        self.assertTrue(transformation.stability_relation(d_in, d_out))

    @parameterized.expand(
        [
            (
                (
                    "Input metric must be IfGroupedBy with an inner metric of"
                    " SymmetricDifference to use AddRemoveIDs as the output metric"
                ),
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                "A",
                True,
            ),
            (
                re.escape(
                    "Input metric must have only a single grouping column (i.e. "
                    "an ID column) to use AddRemoveIDs as the output metric, but found "
                )
                + r"(\{'A', 'B'\}|\{'B', 'A'\})",
                NumpyIntegerDomain(),
                IfGroupedBy(["A", "B"], SymmetricDifference()),
                "A",
                True,
            ),
        ]
    )
    def test_invalid_arguments_raises_error(
        self,
        error_regex: str,
        input_domain: DictDomain,
        input_metric: DictMetric,
        key: Any,
        use_add_remove_ids: bool,
    ):
        """Tests that CreateDictFromValue raises errors appropriately."""
        with self.assertRaisesRegex(ValueError, error_regex):
            CreateDictFromValue(
                input_domain=input_domain,
                input_metric=input_metric,
                key=key,
                use_add_remove_ids=use_add_remove_ids,
            )


class TestDerivedTransformations(unittest.TestCase):
    """Unit tests for derived transformations."""

    def setUp(self) -> None:
        """Set up test."""
        self._halve = create_mock_transformation(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_domain=NumpyFloatDomain(),
            output_metric=AbsoluteDifference(),
            return_value=np.float64(1),
            stability_function_implemented=True,
            stability_function_return_value=2,
        )
        self._triple = create_mock_transformation(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_domain=NumpyIntegerDomain(),
            output_metric=AbsoluteDifference(),
            return_value=np.int64(6),
            stability_function_implemented=True,
            stability_function_return_value=6,
        )

    def test_copy_and_transform_value(self):
        """create_copy_and_transform_value has the expected behavior."""
        data = {"halve_me": np.int64(2), "ignore_me": "nothing to see here"}
        copy_and_transform_value = create_copy_and_transform_value(
            input_domain=DictDomain(
                {"halve_me": NumpyIntegerDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
            input_metric=DictMetric(
                {"halve_me": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
            key="halve_me",
            new_key="halved",
            transformation=self._halve,
            hint=lambda d_in, _: d_in * 2,
        )
        self.assertEqual(
            copy_and_transform_value.input_domain,
            DictDomain(
                {"halve_me": NumpyIntegerDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
        )
        self.assertEqual(
            copy_and_transform_value.input_metric,
            DictMetric(
                {"halve_me": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
        )
        self.assertEqual(
            copy_and_transform_value.output_domain,
            DictDomain(
                {
                    "halve_me": NumpyIntegerDomain(),
                    "ignore_me": NumpyIntegerDomain(),
                    "halved": NumpyFloatDomain(),
                }
            ),
        )
        self.assertEqual(
            copy_and_transform_value.output_metric,
            DictMetric(
                {
                    "halve_me": AbsoluteDifference(),
                    "ignore_me": AbsoluteDifference(),
                    "halved": AbsoluteDifference(),
                }
            ),
        )
        self.assertTrue(
            copy_and_transform_value.stability_relation(
                {"halve_me": 5, "ignore_me": 1},
                {"halve_me": 5, "ignore_me": 1, "halved": 10},
            )
        )
        self.assertEqual(
            copy_and_transform_value(data),
            {
                "halve_me": np.int64(2),
                "ignore_me": "nothing to see here",
                "halved": np.float64(1),
            },
        )

    def test_create_copy_and_transform_value_invalid_key(self):
        """Raises an error if key is not in the domain."""
        with self.assertRaisesRegex(DomainKeyError, "key .* is not in the domain"):
            create_copy_and_transform_value(
                input_domain=DictDomain(
                    {
                        "halve_me": NumpyIntegerDomain(),
                        "ignore_me": NumpyIntegerDomain(),
                    }
                ),
                input_metric=DictMetric(
                    {
                        "halve_me": AbsoluteDifference(),
                        "ignore_me": AbsoluteDifference(),
                    }
                ),
                key="INVALID_KEY",
                new_key="halved",
                transformation=self._halve,
                hint=lambda d_in, _: d_in * 2,
            )

    def test_create_copy_and_transform_value_new_key_already_exists(self):
        """Raises an error if new_key is already in the domain."""
        with self.assertRaisesRegex(ValueError, "new_key is already in the domain"):
            create_copy_and_transform_value(
                input_domain=DictDomain(
                    {
                        "halve_me": NumpyIntegerDomain(),
                        "ignore_me": NumpyIntegerDomain(),
                    }
                ),
                input_metric=DictMetric(
                    {
                        "halve_me": AbsoluteDifference(),
                        "ignore_me": AbsoluteDifference(),
                    }
                ),
                key="halve_me",
                new_key="ignore_me",
                transformation=self._halve,
                hint=lambda d_in, _: d_in * 2,
            )

    def test_create_rename(self):
        """create_rename has the expected behavior."""
        data = {"rename_me": np.int64(2), "ignore_me": "nothing to see here"}
        rename = create_rename(
            input_domain=DictDomain(
                {"rename_me": NumpyIntegerDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
            input_metric=DictMetric(
                {"rename_me": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
            key="rename_me",
            new_key="renamed",
        )
        self.assertEqual(
            rename.input_domain,
            DictDomain(
                {"rename_me": NumpyIntegerDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
        )
        self.assertEqual(
            rename.input_metric,
            DictMetric(
                {"rename_me": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
        )
        self.assertEqual(
            rename.output_domain,
            DictDomain(
                {"renamed": NumpyIntegerDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
        )
        self.assertEqual(
            rename.output_metric,
            DictMetric(
                {"renamed": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
        )
        self.assertTrue(
            rename.stability_relation(
                {"rename_me": 5, "ignore_me": 1}, {"renamed": 5, "ignore_me": 1}
            )
        )
        self.assertEqual(
            rename(data), {"renamed": np.int64(2), "ignore_me": "nothing to see here"}
        )

    def test_create_rename_invalid_key(self):
        """Raises an error if key is not in the domain."""
        with self.assertRaisesRegex(DomainKeyError, "key .* is not in the domain"):
            create_rename(
                input_domain=DictDomain(
                    {
                        "rename_me": NumpyIntegerDomain(),
                        "ignore_me": NumpyIntegerDomain(),
                    }
                ),
                input_metric=DictMetric(
                    {
                        "rename_me": AbsoluteDifference(),
                        "ignore_me": AbsoluteDifference(),
                    }
                ),
                key="INVALID_KEY",
                new_key="renamed",
            )

    def test_create_rename_new_key_already_exists(self):
        """Raises an error if new_key is already in the domain."""
        with self.assertRaisesRegex(ValueError, "new_key is already in the domain"):
            create_rename(
                input_domain=DictDomain(
                    {
                        "rename_me": NumpyIntegerDomain(),
                        "ignore_me": NumpyIntegerDomain(),
                    }
                ),
                input_metric=DictMetric(
                    {
                        "rename_me": AbsoluteDifference(),
                        "ignore_me": AbsoluteDifference(),
                    }
                ),
                key="rename_me",
                new_key="ignore_me",
            )

    def test_create_apply_dict_of_transformations(self):
        """create_apply_dict_of_transformations has the expected behavior."""
        data = np.int64(2)
        apply_dict_of_transformations = create_apply_dict_of_transformations(
            transformation_dict={"halved": self._halve, "tripled": self._triple},
            hint_dict={
                "halved": lambda d_in, _: 2 * d_in,
                "tripled": lambda d_in, _: 3 * d_in,
            },
        )
        self.assertEqual(
            apply_dict_of_transformations.input_domain, NumpyIntegerDomain()
        )
        self.assertEqual(
            apply_dict_of_transformations.input_metric, AbsoluteDifference()
        )
        self.assertEqual(
            apply_dict_of_transformations.output_domain,
            DictDomain({"halved": NumpyFloatDomain(), "tripled": NumpyIntegerDomain()}),
        )
        self.assertEqual(
            apply_dict_of_transformations.output_metric,
            DictMetric(
                {"halved": AbsoluteDifference(), "tripled": AbsoluteDifference()}
            ),
        )
        self.assertTrue(
            apply_dict_of_transformations.stability_relation(
                5, {"halved": 10, "tripled": 15}
            )
        )
        self.assertEqual(
            apply_dict_of_transformations(data),
            {"halved": np.float64(1), "tripled": np.int64(6)},
        )

    def test_create_apply_dict_of_transformations_empty_dict(self):
        """Raises an error if the dict empty."""
        with self.assertRaisesRegex(ValueError, "transformation_dict cannot be empty"):
            create_apply_dict_of_transformations({}, {})

    def test_create_apply_dict_of_transformations_mismatched_keys(self):
        """Raises error is raised if the keys don't match."""
        with self.assertRaisesRegex(
            ValueError, "transformation_dict and hint_dict must have the same keys"
        ):
            create_apply_dict_of_transformations(
                {"key1": create_mock_transformation()}, {"0": lambda d_in, __: d_in}
            )

    def test_create_transform_value(self):
        """create_transform_value has the expected behavior."""
        data = {"halve_me": np.int64(2), "ignore_me": "nothing to see here"}
        transform_value = create_transform_value(
            input_domain=DictDomain(
                {"halve_me": NumpyIntegerDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
            input_metric=DictMetric(
                {"halve_me": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
            key="halve_me",
            transformation=self._halve,
            hint=lambda d_in, _: d_in * 2,
        )
        self.assertEqual(
            transform_value.input_domain,
            DictDomain(
                {"halve_me": NumpyIntegerDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
        )
        self.assertEqual(
            transform_value.input_metric,
            DictMetric(
                {"halve_me": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
        )
        self.assertEqual(
            transform_value.output_domain,
            DictDomain(
                {"halve_me": NumpyFloatDomain(), "ignore_me": NumpyIntegerDomain()}
            ),
        )
        self.assertEqual(
            transform_value.output_metric,
            DictMetric(
                {"halve_me": AbsoluteDifference(), "ignore_me": AbsoluteDifference()}
            ),
        )
        self.assertTrue(
            transform_value.stability_relation(
                {"halve_me": 5, "ignore_me": 1}, {"halve_me": 10, "ignore_me": 1}
            )
        )
        self.assertEqual(
            transform_value(data),
            {"halve_me": np.float64(1), "ignore_me": "nothing to see here"},
        )

    def test_create_transform_value_invalid_key(self):
        """Raises an error if key is not in the domain."""
        with self.assertRaisesRegex(DomainKeyError, "key .* is not in the domain"):
            create_transform_value(
                input_domain=DictDomain(
                    {
                        "halve_me": NumpyIntegerDomain(),
                        "ignore_me": NumpyIntegerDomain(),
                    }
                ),
                input_metric=DictMetric(
                    {
                        "halve_me": AbsoluteDifference(),
                        "ignore_me": AbsoluteDifference(),
                    }
                ),
                key="INVALID_KEY",
                transformation=self._halve,
                hint=lambda d_in, _: d_in * 2,
            )

    def test_create_transform_all_values(self):
        """create_transform_all_values has the expected behavior."""
        data = {"halve_me": np.int64(2), "triple_me": np.int64(2)}
        transform_all_values = create_transform_all_values(
            transformation_dict={
                "halve_me": self._halve,
                "triple_me": self._triple,
            },
            hint_dict={
                "halve_me": lambda d_in, _: d_in * 2,
                "triple_me": lambda d_in, _: d_in * 3,
            },
        )
        self.assertEqual(
            transform_all_values.input_domain,
            DictDomain(
                {"halve_me": NumpyIntegerDomain(), "triple_me": NumpyIntegerDomain()}
            ),
        )
        self.assertEqual(
            transform_all_values.input_metric,
            DictMetric(
                {"halve_me": AbsoluteDifference(), "triple_me": AbsoluteDifference()}
            ),
        )
        self.assertEqual(
            transform_all_values.output_domain,
            DictDomain(
                {"halve_me": NumpyFloatDomain(), "triple_me": NumpyIntegerDomain()}
            ),
        )
        self.assertEqual(
            transform_all_values.output_metric,
            DictMetric(
                {"halve_me": AbsoluteDifference(), "triple_me": AbsoluteDifference()}
            ),
        )
        self.assertTrue(
            transform_all_values.stability_relation(
                {"halve_me": 5, "triple_me": 5}, {"halve_me": 10, "triple_me": 15}
            )
        )
        self.assertEqual(
            transform_all_values(data),
            {"halve_me": np.float64(1), "triple_me": np.int64(6)},
        )

    def test_create_transform_all_values_empty(self):
        """Raises an error if the dict empty."""
        with self.assertRaisesRegex(ValueError, "transformation_dict cannot be empty"):
            create_transform_all_values({}, {})

    def test_create_transform_all_values_mismatched_keys(self):
        """Raises error is raised if the keys don't match."""
        with self.assertRaisesRegex(
            ValueError, "transformation_dict and hint_dict must have the same keys"
        ):
            create_transform_all_values(
                {"key1": create_mock_transformation()}, {"0": lambda d_in, __: d_in}
            )
