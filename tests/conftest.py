"""Command-line options of these tests (the shared code in cosmolike_core).

pytest reads a file named conftest.py from the folders above the tests
it collects, so this file must stay in tests/. It registers two options
of the comparison sweeps:

  --high=1        repeats the CFASTPT-vs-FASTPT sweep (test_fastpt.py) at
                  the high-accuracy settings instead of the frozen
                  defaults (--high=0, the default);
  --mask=<name>   runs the CFASTPT-vs-FASTPT and Halofit-vs-EE2 sweeps
                  (test_fastpt.py, test_nonlinear.py) under another
                  scale-cut mask: "frozen" (the default) keeps the mask
                  of each frozen configuration, "ones" keeps every entry.

The implementation lives in cosmolike_core/cocoa_testing.py
(conftest_addoption, conftest_configure), bound here the same way
cocoa_test_utils.py binds the test harness; the --mask choices come from
this project's harness (its fastpt_masks tuple).
"""

import os
import sys

# The tests folder is not a package; put it on the import path so the
# project shim resolves no matter where pytest was launched from (the
# shim itself puts cosmolike_core on the path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


def pytest_addoption(parser):
    """Register --high and --mask on pytest's parser.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_addoption.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the options become readable through config.getoption.
    """
    u._cct.conftest_addoption(parser, u._H.fastpt_masks)


def pytest_configure(config):
    """Copy the option values where the test classes read them.

    The values land in the environment variables COCOA_FASTPT_HIGH and
    COCOA_FASTPT_MASK: the tests are unittest.TestCase classes, whose
    methods cannot receive pytest fixtures, so they read the environment
    (with the defaults "0" and "frozen", which a run outside pytest
    also gets). The implementation lives in
    cocoa_testing.conftest_configure.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains the variables.
    """
    u._cct.conftest_configure(config)
