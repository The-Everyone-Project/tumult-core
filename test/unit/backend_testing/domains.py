"""Backend-neutral domain construction for the parity harness.

This module is part of the frozen harness API; see
:mod:`test.unit.backend_testing` for the freeze contract.

Nothing here is implemented yet. :func:`domain_for` is a placeholder whose
*signature* is frozen now, so that the suites written against this harness can
call it as soon as the pandas domains land, without a second round of edits to
every one of them.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.backend_testing.conversion import BackendLike
from typing import Any


def domain_for(schema: Any, backend: BackendLike) -> Any:
    """Returns the domain a backend's frames with the given schema belong to.

    This is the domain counterpart of
    :func:`~test.unit.backend_testing.conversion.df_for`: a test describes its
    data's schema once and gets
    :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain` or the
    pandas equivalent depending on which backend it is running against, so that
    one test body can build a transformation or measurement for either.

    .. warning::
        Not implemented. Calling this raises
        :class:`NotImplementedError`.

        TODO: Implement this once the pandas domains land (work package C1,
        ``src/tmlt/core/domains/pandas_domains.py``). Deliberately left as a
        placeholder here: C1 is in flight in a parallel branch and owns that
        file, so this package must not import from it yet. The signature is
        frozen as of this package, so implementing it is a change to this
        function's body and nothing else.

    Args:
        schema: The frame's schema. The intended form is a mapping from column
            name to column type; a :class:`~pyspark.sql.types.StructType` is
            expected to be accepted too. Deliberately typed as ``Any`` until
            C1 fixes which of the two the pandas domains are built from --
            narrowing an annotation later is compatible, widening one is not.
        backend: The backend whose domain is wanted.

    Returns:
        The :class:`~tmlt.core.domains.base.Domain` for ``schema`` under
        ``backend``. Typed as ``Any`` rather than ``Domain`` so that this
        module does not import the domains package while C1 is reshaping it.

    Raises:
        NotImplementedError: Always, for now.
    """
    raise NotImplementedError(
        "domain_for is a placeholder: the pandas domains (work package C1) "
        f"have not landed yet, so no domain can be built for the "
        f"{backend.name} backend. Its signature is frozen; only its body is "
        "pending."
    )
