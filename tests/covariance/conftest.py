"""Shared set-up of the covariance tests: import path and build check.

The covariance tests import the compiled interface directly, so this
file puts projects/roman_kl/interface on the module search path
(parents[2] of this file is the project folder). The covariance code is
compiled into the interface unless IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE
is set; such a build has no covariance bindings (its has_covariance
attribute is False), and then every test of this folder is skipped
with the instruction to rebuild.
"""

from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_roman_kl_interface as ci


@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip the covariance tests when the interface lacks covariance code.

    @pytest.fixture(scope="session", autouse=True) makes pytest run this
    function once, before the first test of this folder, without any test
    asking for it. pytest.skip stops it there and marks the tests skipped,
    with a message that says how to rebuild.

    Returns:
      nothing when the interface carries the covariance bindings.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
