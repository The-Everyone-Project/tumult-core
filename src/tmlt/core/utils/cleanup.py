"""Cleanup functions for Tumult Core."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import atexit
import re
from typing import List

from pyspark.sql import SparkSession

from tmlt.core.utils.configuration import Config


def _cleanup_temp() -> None:
    """Cleanup the temporary table, if a Spark session is running.

    This asks for the *active* session rather than calling
    ``SparkSession.builder.getOrCreate()``. It is registered as an ``atexit``
    hook, and getOrCreate does what its name says: it would start a JVM on the
    way out of every process that imported this module, including the ones --
    a pandas-only pipeline, a script that only touched
    :mod:`tmlt.core.utils.arb` -- that never had a Spark session to clean up
    after. A process with no active session has no temporary database of ours,
    so there is nothing to do.
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        return

    spark.sql(f"DROP DATABASE IF EXISTS `{Config.temp_db_name()}` CASCADE")


def cleanup() -> None:
    """Cleanup Core's temporary table.

    If you call ``spark.stop()``, you should call this function first: it
    cleans up the *active* Spark session's temporary table, and after
    ``spark.stop()`` there is no active session and nothing happens.
    """
    _cleanup_temp()


def remove_all_temp_tables() -> None:
    """Remove all temporary tables that Core has created.

    This will remove all temporary tables in the current Spark
    data warehouse.
    """
    spark = SparkSession.builder.getOrCreate()
    pattern = re.compile(r"tumult_temp_\d{8}_\d{6}_(\d|a-f)*")
    dbs_to_remove: List[str] = []
    for db in spark.catalog.listDatabases():
        if pattern.match(db.name):
            dbs_to_remove.append(db.name)

    for db_name in dbs_to_remove:
        spark.sql(f"DROP DATABASE `{db_name}` CASCADE")


atexit.register(_cleanup_temp)
