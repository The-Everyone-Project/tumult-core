"""Unit tests for :mod:`tmlt.core.utils.cleanup`."""

import subprocess
import sys
from pathlib import Path
from random import choice, randint
from string import ascii_letters, digits
from textwrap import dedent
from uuid import uuid4

from tmlt.core.utils.cleanup import cleanup, remove_all_temp_tables
from tmlt.core.utils.configuration import Config
from tmlt.core.utils.testing import PySparkTest

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

_IMPORT_CLEANUP_AND_EXIT = '''
"""Imports the cleanup module with a JVM launch forbidden, then exits."""

import pyspark.context
import pyspark.java_gateway


def forbidden(*args, **kwargs):
    """Stands in for launch_gateway.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError("a JVM was started")


# getOrCreate() calls the launch_gateway name that pyspark.context imported,
# not the one pyspark.java_gateway defines, so both bindings have to go.
pyspark.java_gateway.launch_gateway = forbidden
pyspark.context.launch_gateway = forbidden

import tmlt.core.utils.cleanup  # noqa: E402,F401
'''


def test_atexit_hook_does_not_start_a_jvm(tmp_path: Path) -> None:
    """Importing the module does not start a JVM at interpreter exit.

    ``_cleanup_temp`` is registered with ``atexit``. When it asked for its Spark
    session with ``getOrCreate`` it built one -- JVM and all -- on the way out of
    every process that had imported the module, including processes that never
    touched Spark and so cannot have left a temporary database behind.

    This runs in a subprocess because the hook only fires when an interpreter
    shuts down, and it inspects stderr rather than the exit status because an
    exception raised in an ``atexit`` hook is printed and then ignored: the
    process still exits 0.

    Args:
        tmp_path: Directory to write the subprocess' script into.
    """
    script = tmp_path / "import_cleanup.py"
    script.write_text(dedent(_IMPORT_CLEANUP_AND_EXIT))

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    )

    assert "a JVM was started" not in result.stderr, result.stderr
    assert "atexit" not in result.stderr, result.stderr


class TestCleanup(PySparkTest):
    """TestCase for cleanup functions."""

    def _generate_fake_data(self):
        """Generate some arbitrary data (for tests)."""
        rows = []
        size = randint(1, 10)
        for _ in range(size):
            s = ""
            for _ in range(randint(1, 10)):
                s += choice(ascii_letters + digits)
            rows.append((s, randint(-100, 100)))
        return self.spark.createDataFrame(rows, schema=["name", "number"])

    def _get_warehouse_path(self) -> Path:
        """Get a Path object pointing to Spark's warehouse directory."""
        config_path = self.spark.conf.get("spark.sql.warehouse.dir")
        if config_path is None:
            raise RuntimeError("Unable to get config path")
        if config_path[0:5] == "file:":
            config_path = config_path[5:]
        return Path(config_path).resolve()

    def setUp(self):
        """Setup."""
        prev_db = self.spark.catalog.currentDatabase()
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS `{Config.temp_db_name()}`;")
        self.spark.catalog.setCurrentDatabase(Config.temp_db_name())
        for i in range(3):
            self._generate_fake_data().write.saveAsTable(f"table{i}")

        self.spark.catalog.setCurrentDatabase(prev_db)

    @staticmethod
    def _recursive_remove(p: Path):
        """Recursively remove a path (just like `rm -r`)."""
        if not p.is_dir():
            p.unlink()
        for f in p.iterdir():
            TestCleanup._recursive_remove(f)
        p.rmdir()

    def tearDown(self):
        """Cleanup."""
        self.spark.sql(f"DROP DATABASE IF EXISTS `{Config.temp_db_name()}` CASCADE;")
        for d in self._get_warehouse_path().glob(f"*{Config.temp_db_name()}*"):
            self._recursive_remove(d)
        for d in self._get_warehouse_path().glob("*tumult_temp*"):
            self._recursive_remove(d)

    def test_cleanup(self):
        """Cleanup cleans up all temporary tables."""
        cleanup()
        self.assertNotIn(
            Config.temp_db_name(),
            [db.name for db in self.spark.catalog.listDatabases()],
        )
        # Check to ensure that the relevant directory is also gone or empty
        # The warehouse may or may not have a directory for the database ...
        for d in self._get_warehouse_path().glob(f"*{Config.temp_db_name}*"):
            # ... but if it does, that directory should be empty
            self.assertEqual(len(list(d.iterdir())), 0)

        # Cleanup should also work when the database doesn't exist
        cleanup()
        self.assertNotIn(
            "db_that_does_not_exist",
            [db.name for db in self.spark.catalog.listDatabases()],
        )
        self.assertNotIn(
            Config.temp_db_name(),
            [db.name for db in self.spark.catalog.listDatabases()],
        )
        # Check to ensure that the relevant directory is also gone or empty
        # The warehouse may or may not have a directory for the database ...
        for d in self._get_warehouse_path().glob(f"*{Config.temp_db_name}*"):
            # ... but if it does, that directory should be empty
            self.assertEqual(len(list(d.iterdir())), 0)

    def test_remove_all_temp_tables(self):
        """Remove_all_temp_tables removes all tables."""
        # It should work when there's just one table
        remove_all_temp_tables()
        self.assertNotIn(
            Config.temp_db_name(),
            [db.name for db in self.spark.catalog.listDatabases()],
        )
        # Check to ensure that the relevant directory is also gone or empty
        # The warehouse may or may not have a directory for the database ...
        for d in self._get_warehouse_path().glob(f"*{Config.temp_db_name}*"):
            # ... but if it does, that directory should be empty
            self.assertEqual(len(list(d.iterdir())), 0)

        # Make some fake temporary tables
        fake_dbs = [
            f"tumult_temp_19700101_000000_{uuid4().hex}",
            f"tumult_temp_20380119_031408_{uuid4().hex}",
            f"tumult_temp_21060207_062815_{uuid4().hex}",
            Config.temp_db_name(),
        ]
        prev_db = self.spark.catalog.currentDatabase()
        for fake_db in fake_dbs:
            self.spark.sql(f"CREATE DATABASE IF NOT EXISTS `{fake_db}`;")
            self.spark.catalog.setCurrentDatabase(fake_db)
            for i in range(3):
                self._generate_fake_data().write.saveAsTable(f"table{i}")
        self.spark.catalog.setCurrentDatabase(prev_db)
        # remove_all should work when all these tables exist
        remove_all_temp_tables()
        for fake_db in fake_dbs:
            self.assertNotIn(
                fake_db, [db.name for db in self.spark.catalog.listDatabases()]
            )
            # Check to ensure that the relevant directory is also gone or empty
            # The warehouse may or may not have a directory for the database ...
            for d in self._get_warehouse_path().glob(f"*{fake_db}*"):
                # ... but if it does, that directory should be empty
                self.assertEqual(len(list(d.iterdir())), 0)

        # Remove_all should work when no temp tables exist
        remove_all_temp_tables()
        self.assertNotIn(
            Config.temp_db_name(),
            [db.name for db in self.spark.catalog.listDatabases()],
        )
