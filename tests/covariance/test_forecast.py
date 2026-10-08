"""Check the roman_kl covariance adapter against the shared covariance code.

roman_kl_covariance.py (the adapter: the survey choices of this project)
is exercised by cocoa_covariance_testing.check_project_forecast from
cosmolike_core, with small numerical settings and three measured
observables (one shear, one galaxy-shear and one clustering pair, two
bins each). It checks the full layout sizes, the accuracy controls, that
the notebook wrappers and the production bindings give bitwise equal
components in real and Fourier space, that one and eight OpenMP threads
give bitwise equal matrices, that every matrix is finite and symmetric
with total = G + SSC + cNG and a positive-definite total, and that the
saved archive reads back intact. It does not certify the convergence of
the survey covariance.

pytest.ini selects only tests/data_vector by default, so name this
folder explicitly (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_kl/tests/covariance
"""

from pathlib import Path
import sys

# project = projects/roman_kl; cosmolike_core holds the shared check, and
# interface/ and covariance/ hold the compiled module and the adapter
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_roman_kl_interface as ci
import roman_kl_covariance as survey


def test_forecast_adapter(tmp_path):
    """Run the shared adapter check on this project.

    expected_sizes = (3300, 2200): the full real-space layout (55 xi_+
    and 55 xi_- shear blocks, 45 galaxy-shear and 10 clustering blocks
    of 20 angular bins) and the Fourier layout (55 + 45 + 10 blocks of
    20 bands).

    Arguments:
      tmp_path = a temporary folder pytest creates for this test (a
                 fixture: pytest passes it by the argument's name); the
                 saved archives go there.
    """
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(3300, 2200), directory=tmp_path,
    )
