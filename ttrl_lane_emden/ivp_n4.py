"""Lane--Emden n=4 IVP verifier and reward shaping.

Unlike n=1, the physical n=4 Lane--Emden solution has no simple elementary
closed form.  We therefore certify *approximate IVP solutions* using only
mathematical information derived from the ODE and its initial conditions:

    y'' + 2/x y' + y^4 = 0,   y(0)=1, y'(0)=0.

The neural model may emit two symbolic integration constants.  We fit those
constants to the unique physical IVP using weighted multi-point anchor conditions
obtained from a high-accuracy numerical integration, then score the resulting
expression on disjoint collocation/reference grids.
"""
from __future__ import annotations

import math
import signal
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

class CandidateVerificationTimeout(BaseException):
    pass


@contextmanager
def _candidate_time_limit(seconds: float):
    """Hard wall-clock timeout for one CPU-side candidate verification."""
    seconds = float(seconds or 0.0)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise CandidateVerificationTimeout(
            f"candidate verification exceeded {seconds:g}s"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


from ttrl_lane_emden.core import (
    _coefficient_symbols,
    generated_to_candidates,
    ids_to_sympy,
)


@dataclass
class LaneEmdenN4Reference:
    x_min: float = 0.05
    x_max: float = 14.8
    rtol: float = 1e-11
    atol: float = 1e-13

    def __post_init__(self):
        # Start very close to the regular singular point using the analytic
        # Lane--Emden n=4 Taylor series.
        eps = 1e-5
        x = eps
        # a[k+1] = -[z**k](sum_j a[j]*z**j)**4 / ((2*k+2)*(2*k+3)).
        # Exact rational coefficients through x**10, independently recomputed.
        coefficients = np.array([
            1.0 / 1.0,
            -1.0 / 6.0,
            1.0 / 30.0,
            -1.0 / 140.0,
            43.0 / 27216.0,
            -26641.0 / 74844000.0,
        ], dtype=np.float64)
        y0 = np.polynomial.polynomial.polyval(x*x, coefficients)
        dy0 = 2.0*x*np.polynomial.polynomial.polyval(
            x*x, np.arange(1, len(coefficients))*coefficients[1:]
        )

        def rhs(t, z):
            y, yp = z
            return [yp, -2.0 * yp / t - y**4]

        def first_zero(t, z):
            return z[0]

        first_zero.direction = -1

