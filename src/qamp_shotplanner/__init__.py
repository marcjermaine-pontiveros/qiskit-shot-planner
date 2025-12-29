"""QAMP Shot Planner - Budgeting circuit runs using concentration bounds."""

__version__ = "0.1.0"

from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner
from qamp_shotplanner.swaptest.circuit import swap_test_1qubit
from qamp_shotplanner.swaptest.planning import plan_shots_for_swap_fidelity
from qamp_shotplanner.swaptest.run_estimator import run_swap_fidelity_estimator
from qamp_shotplanner.validation.coverage import coverage_validation_swap, CoverageStats

__all__ = [
    "HoeffdingPlanner",
    "swap_test_1qubit",
    "plan_shots_for_swap_fidelity",
    "run_swap_fidelity_estimator",
    "coverage_validation_swap",
    "CoverageStats",
]