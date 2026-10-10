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
        first_zero.terminal = False
        # Continue slightly past the physical surface for root diagnostics only.
        # Reward/reference grids remain within x_max on the positive branch.
        self.integration_end = max(float(self.x_max), 17.0)
        self.sol = solve_ivp(
            rhs,
            (eps, self.integration_end),
            [y0, dy0],
            method="DOP853",
            dense_output=True,
            events=first_zero,
            rtol=self.rtol,
            atol=self.atol,
            max_step=0.05,
        )
        if not self.sol.success:
            raise RuntimeError(f"Lane--Emden n=4 reference integration failed: {self.sol.message}")

        if not len(self.sol.t_events[0]):
            raise RuntimeError("Lane--Emden reference did not reach its first zero")
        self.first_zero = float(self.sol.t_events[0][0])
        self._eps = eps
        self._coefficients = coefficients

    def _values(self, xs, derivative=False):
        xs = np.asarray(xs, dtype=np.float64)
        if not np.all(np.isfinite(xs)) or np.any(xs < 0) or np.any(xs > self.integration_end):
            raise ValueError("reference points must be finite and inside the integration interval")
        values = np.asarray(self.sol.sol(np.maximum(xs, self._eps))[int(derivative)], dtype=np.float64)
        if derivative:
            central = 2.0*xs*np.polynomial.polynomial.polyval(
                xs*xs, np.arange(1, len(self._coefficients))*self._coefficients[1:]
            )
        else:
            central = np.polynomial.polynomial.polyval(xs*xs, self._coefficients)
        return np.where(xs < self._eps, central, values)

    def y(self, xs):
        return self._values(xs)

    def yp(self, xs):
        return self._values(xs, derivative=True)


@dataclass
class IVPRewardInfo:
    reward: float
    valid_parse: bool
    certified_ivp: bool
    elite_eligible: bool
    ode_rel_mse: float
    ref_nrmse: float
    anchor_rmse: float
    expression: Optional[sp.Expr]
    fitted_expression: Optional[sp.Expr]
    fitted_coefficients: Dict[str, float]
    error: Optional[str] = None

    # Compatibility aliases used by some generic logging helpers.
    @property
    def numerical_mse(self):
        return self.ode_rel_mse

    @property
    def exact_residual_zero(self):
        return False

    @property
    def generality_rank(self):
        return 0

    @property
    def verified_general(self):
        return False


class LaneEmdenN4Verifier:
    def __init__(
        self,
        env,
        x_anchor: float = 0.10,          # kept for backward compatibility (first point)
        x_max: float = 14.8,
        n_ode_points: int = 144,
        n_ref_points: int = 192,
        certify_ode_rel: float = 5e-2,
        certify_ref_nrmse: float = 3e-2,
        certify_anchor_rmse: float = 2e-3,
        elite_ode_rel: float = 1.2e-1,
        elite_ref_nrmse: float = 1.2e-1,
        elite_anchor_rmse: float = 1e-2,
        max_coeffs: int = 2,
        candidate_timeout_s: float = 3.0,
        # ---- new multi-anchor settings ----
        anchor_points: Optional[Sequence[float]] = None,
        anchor_weights: Optional[Sequence[float]] = None,
    ):
        self.env = env
        self.x = env.local_dict["x"]
        self.f = env.local_dict["f"]
        self.x_anchor = float(x_anchor)
        self.x_max = float(x_max)
        self.max_coeffs = int(max_coeffs)
        self.certify_ode_rel = float(certify_ode_rel)
        self.certify_ref_nrmse = float(certify_ref_nrmse)
        self.certify_anchor_rmse = float(certify_anchor_rmse)
        self.elite_ode_rel = float(elite_ode_rel)
        self.elite_ref_nrmse = float(elite_ref_nrmse)
        self.elite_anchor_rmse = float(elite_anchor_rmse)
        self.candidate_timeout_s = float(candidate_timeout_s)
        if not 0 < self.x_anchor < self.x_max:
            raise ValueError("require 0 < x_anchor < x_max")
        if n_ode_points < 2 or n_ref_points < 2:
            raise ValueError("require at least two ODE and reference grid points")
        self.reference = LaneEmdenN4Reference(x_min=0.05, x_max=x_max)

        # ---------- multi-point anchor setup ----------
        if anchor_points is None:
            # Recommended default set (near-origin heavy)
            self.anchor_x = np.array([0.05, 0.15, 0.40, 1.00, 2.00], dtype=np.float64)
        else:
            self.anchor_x = np.asarray(anchor_points, dtype=np.float64)

        if anchor_weights is None:
            # Higher weight near the origin
            self.anchor_w = np.array([1.0, 1.0, 1.0, 0.7, 0.4], dtype=np.float64)
            if anchor_points is not None:
                self.anchor_w = np.exp(-self.anchor_x / 0.8)
        else:
            self.anchor_w = np.asarray(anchor_weights, dtype=np.float64)
        if (self.anchor_x.ndim != 1 or not len(self.anchor_x) or
            not np.all(np.isfinite(self.anchor_x)) or
            np.any(self.anchor_x < 0) or np.any(self.anchor_x > self.x_max)):
            raise ValueError("anchor_points must be a nonempty finite vector inside [0, x_max]")
        if (self.anchor_w.shape != self.anchor_x.shape or
            not np.all(np.isfinite(self.anchor_w)) or np.any(self.anchor_w < 0) or
            np.sum(self.anchor_w) <= 0):
            raise ValueError("anchor_weights must match anchors, be nonnegative, and have positive sum")
        self.anchor_w = self.anchor_w / np.sum(self.anchor_w)   # normalise

        # Pre-compute reference values at the anchor points
        self.anchor_y  = self.reference.y(self.anchor_x)
        self.anchor_yp = self.reference.yp(self.anchor_x)

        # Keep the old single-point target for any legacy code that might still look at it
        self.anchor_target = np.array([
            float(self.reference.y([self.x_anchor])[0]),
            float(self.reference.yp([self.x_anchor])[0]),
        ], dtype=np.float64)

        # ---------- rest of the original initialisation ----------
        y = self.f(self.x)
        yp = sp.diff(y, self.x)
        ypp = sp.diff(y, self.x, 2)
        self.input_terms = (self.x * ypp, 2 * yp, self.x * y**4)
        self.equation = sp.Add(*self.input_terms)

        self.ode_x = np.linspace(self.x_anchor, self.x_max, n_ode_points, dtype=np.float64)
        step = (self.x_max - self.x_anchor) / max(n_ref_points, 1)
        self.ref_x = np.linspace(self.x_anchor + 0.37 * step, self.x_max - 0.19 * step,
                                 n_ref_points, dtype=np.float64)
        self.ref_y = self.reference.y(self.ref_x)

    def _fit_coefficients(self, hyp: sp.Expr):
        coeffs = _coefficient_symbols(self.env, hyp)
        if len(coeffs) > self.max_coeffs:
            raise ValueError(f"candidate has {len(coeffs)} coefficients; max supported is {self.max_coeffs}")
        if not coeffs:
            return hyp, {}, self._anchor_rmse(hyp)

        yp = sp.diff(hyp, self.x)
        fn_y  = sp.lambdify([self.x] + coeffs, hyp, modules=["numpy"])
        fn_yp = sp.lambdify([self.x] + coeffs, yp,  modules=["numpy"])

        n_pts = len(self.anchor_x)
        # residual vector length = 2 * number of anchors  (y and y' at each point)
        def residual(cvals):
            try:
                res = np.empty(2 * n_pts, dtype=np.float64)
                for i, xi in enumerate(self.anchor_x):
                    yv = complex(fn_y(xi, *cvals))
                    dv = complex(fn_yp(xi, *cvals))
                    if (not np.isfinite(yv.real) or not np.isfinite(yv.imag) or
                        not np.isfinite(dv.real) or not np.isfinite(dv.imag) or
                        abs(yv.imag) > 1e-7 or abs(dv.imag) > 1e-7):
                        return np.full(2 * n_pts, 1e4, dtype=np.float64)
                    # weighted residuals
                    w = math.sqrt(self.anchor_w[i])
                    res[2*i]     = w * (yv.real - self.anchor_y[i])
                    res[2*i + 1] = w * (dv.real - self.anchor_yp[i])
                return res
            except Exception:
                return np.full(2 * n_pts, 1e4, dtype=np.float64)

        starts = [
            np.zeros(len(coeffs)),
            np.ones(len(coeffs)),
            -np.ones(len(coeffs)),
        ]
        if len(coeffs) == 2:
            starts += [np.array([1.0, -1.0]), np.array([-1.0, 1.0])]

        best = None
        for s in starts:
            try:
                out = least_squares(
                    residual,
                    s,
                    bounds=(-20.0, 20.0),
                    max_nfev=120,          # a bit more budget for multi-point
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                )
                err = float(np.sqrt(np.mean(np.square(residual(out.x)))))
                if best is None or err < best[0]:
                    best = (err, out.x.copy())
            except Exception:
                continue

        if best is None:
            raise ValueError("coefficient fitting failed")

        coeff_map = {c: float(v) for c, v in zip(coeffs, best[1])}
        fitted = hyp.subs(coeff_map)
        return fitted, {str(c): float(v) for c, v in coeff_map.items()}, float(best[0])

    def _anchor_rmse(self, fitted: sp.Expr):
        """RMSE of (y, y') over the multi-point anchor set (weighted)."""
        yp = sp.diff(fitted, self.x)
        try:
            fy  = sp.lambdify(self.x, fitted, modules=["numpy"])
            fyp = sp.lambdify(self.x, yp,     modules=["numpy"])

            errs = []
            for i, xi in enumerate(self.anchor_x):
                yv = complex(fy(xi))
                dv = complex(fyp(xi))
                if (not np.isfinite(yv.real) or not np.isfinite(dv.real) or
                    abs(yv.imag) > 1e-7 or abs(dv.imag) > 1e-7):
                    return 1e6
                w = self.anchor_w[i]
                errs.append(w * (yv.real - self.anchor_y[i])**2)
                errs.append(w * (dv.real - self.anchor_yp[i])**2)
            return float(np.sqrt(np.mean(errs)))
        except Exception:
            return 1e6

    def evaluate_accuracy(self, fitted: sp.Expr, grid=None):
        """
        Pure evaluation diagnostics (never used for reward / training).
        Returns a dict with absolute/relative error tables, max abs error,
        and first-zero error versus the high-accuracy reference.
        """
        if fitted is None:
            return {
                "abs_error_table": [],
                "rel_error_table": [],
                "max_abs_error": 1e6,
                "first_zero_approx": None,
                "first_zero_error": 1e6,
            }

        if grid is None:
            grid = np.linspace(0.5, self.x_max, 12, dtype=np.float64)

        grid = np.asarray(grid, dtype=np.float64)
        ref_y = self.reference.y(grid)
        try:
            fy = sp.lambdify(self.x, fitted, modules=["numpy"])
            pred = np.asarray(fy(grid), dtype=np.complex128)
            if pred.ndim == 0:
                pred = np.full(grid.shape, pred, dtype=np.complex128)
            pred = np.broadcast_to(pred, grid.shape)
            if not np.all(np.isfinite(pred)) or np.max(np.abs(pred.imag)) > 1e-7:
                raise ValueError("non-real or non-finite diagnostic prediction")
            pred = pred.real
        except Exception:
            return {
                "abs_error_table": [],
                "rel_error_table": [],
                "max_abs_error": 1e6,
                "first_zero_approx": None,
                "first_zero_error": 1e6,
            }

        abs_err = np.abs(pred - ref_y)
        rel_err = np.abs(pred - ref_y) / (np.abs(ref_y) + 1e-12)

        abs_table = [(float(x), float(e)) for x, e in zip(grid, abs_err)]
        rel_table = [(float(x), float(e)) for x, e in zip(grid, rel_err)]
        max_abs = float(np.max(abs_err))

        # First zero of the approximate solution
        xi1_ref = self.reference.first_zero
        try:
            xs = np.linspace(0.0, self.reference.integration_end, 4000)
            ys = np.broadcast_to(np.asarray(fy(xs), dtype=np.complex128), xs.shape)
            real_finite = np.isfinite(ys) & (np.abs(ys.imag) <= 1e-7)
            ys = ys.real
            zero_idx = np.where(real_finite[:-1] & real_finite[1:] &
                                (ys[:-1] > 0) & (ys[1:] <= 0))[0]
            if len(zero_idx) > 0:
                i = zero_idx[0]
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[i], ys[i + 1]
                def root_value(t):
                    value = complex(fy(t))
                    if not np.isfinite(value) or abs(value.imag) > 1e-7:
                        raise ValueError("invalid value during root refinement")
                    return value.real
                xi1_approx = brentq(root_value, x0, x1, xtol=1e-12)
                if abs(root_value(xi1_approx)) > 1e-8:
                    raise ValueError("sign change is not a finite zero")
                first_zero_err = abs(xi1_approx - xi1_ref)
            else:
                xi1_approx = None
                first_zero_err = 1e6
        except Exception:
            xi1_approx = None
            first_zero_err = 1e6

        return {
            "abs_error_table": abs_table,
            "rel_error_table": rel_table,
            "max_abs_error": max_abs,
            "first_zero_approx": float(xi1_approx) if xi1_approx is not None else None,
            "first_zero_error": float(first_zero_err),
        }

    def _metrics(self, fitted: sp.Expr):
        yp = sp.diff(fitted, self.x)
        ypp = sp.diff(fitted, self.x, 2)
        terms = (self.x * ypp, 2 * yp, self.x * fitted**4)
        try:
            fterms = [sp.lambdify(self.x, t, modules=["numpy"]) for t in terms]
            arrs = []
            for fn in fterms:
                v = np.asarray(fn(self.ode_x), dtype=np.complex128)
                if v.ndim == 0:
                    v = np.full(self.ode_x.shape, v, dtype=np.complex128)
                v = np.broadcast_to(v, self.ode_x.shape)
                if not np.all(np.isfinite(v)) or np.max(np.abs(v.imag)) > 1e-7:
                    return 1e6, 1e6
                arrs.append(np.clip(v.real, -1e50, 1e50))
            arr = np.stack(arrs, axis=0)
            residual = arr.sum(axis=0)
            den = np.sum(arr * arr, axis=0)
            valid = den > 1e-20
            if not np.any(valid):
                ode_rel = 1e6
            else:
                ode_rel = float(np.mean((residual[valid] ** 2) / np.maximum(den[valid], 1e-30)))

            fy = sp.lambdify(self.x, fitted, modules=["numpy"])
            pred = np.asarray(fy(self.ref_x), dtype=np.complex128)
            if pred.ndim == 0:
                pred = np.full(self.ref_x.shape, pred, dtype=np.complex128)
            pred = np.broadcast_to(pred, self.ref_x.shape)
            if not np.all(np.isfinite(pred)) or np.max(np.abs(pred.imag)) > 1e-7:
                return ode_rel, 1e6
            scale = float(np.sqrt(np.mean(self.ref_y * self.ref_y))) + 1e-12
            ref_nrmse = float(np.sqrt(np.mean((pred.real - self.ref_y) ** 2)) / scale)
            return ode_rel, ref_nrmse
        except Exception:
            return 1e6, 1e6

    def score_expr(self, hyp: sp.Expr) -> IVPRewardInfo:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fitted, coeffs, anchor_rmse = self._fit_coefficients(hyp)
                ode_rel, ref_nrmse = self._metrics(fitted)
            if not all(math.isfinite(v) for v in (anchor_rmse, ode_rel, ref_nrmse)):
                raise ValueError("non-finite IVP metric")

            # Dense reward is based on *joint* normalized constraint violation.
            # This prevents a candidate from receiving a high reward merely by
            # driving one metric (especially ODE residual) extremely small while
            # being badly wrong on the physical IVP.  Each term is normalized by
            # the strict certification threshold, then compressed with log1p.
            ode_ratio = max(ode_rel, 0.0) / max(self.certify_ode_rel, 1e-12)
            ref_ratio = max(ref_nrmse, 0.0) / max(self.certify_ref_nrmse, 1e-12)
            anchor_ratio = max(anchor_rmse, 0.0) / max(self.certify_anchor_rmse, 1e-12)
            joint_penalty = (
                math.log1p(ode_ratio)
                + 2.0 * math.log1p(ref_ratio)
                + 0.5 * math.log1p(anchor_ratio)
            )
            reward = -float(joint_penalty)

            certified = (
                anchor_rmse <= self.certify_anchor_rmse
                and ode_rel <= self.certify_ode_rel
                and ref_nrmse <= self.certify_ref_nrmse
            )
            elite_eligible = (
                anchor_rmse <= self.elite_anchor_rmse
                and ode_rel <= self.elite_ode_rel
                and ref_nrmse <= self.elite_ref_nrmse
            )
            if certified:
                reward += 10.0
            elif elite_eligible:
                reward += 3.0

            return IVPRewardInfo(
                reward=float(reward),
                valid_parse=True,
                certified_ivp=bool(certified),
                elite_eligible=bool(elite_eligible),
                ode_rel_mse=float(ode_rel),
                ref_nrmse=float(ref_nrmse),
                anchor_rmse=float(anchor_rmse),
                expression=hyp,
                fitted_expression=fitted,
                fitted_coefficients=coeffs,
            )
        except CandidateVerificationTimeout:
            raise
        except BaseException as e:
            return IVPRewardInfo(
                reward=-25.0,
                valid_parse=False,
                certified_ivp=False,
                elite_eligible=False,
                ode_rel_mse=1e6,
                ref_nrmse=1e6,
                anchor_rmse=1e6,
                expression=hyp,
                fitted_expression=None,
                fitted_coefficients={},
                error=f"{type(e).__name__}: {e}",
            )

    def score_ids(self, token_ids: Sequence[int]) -> IVPRewardInfo:
        hyp = None
        try:
            with _candidate_time_limit(self.candidate_timeout_s):
                hyp = ids_to_sympy(self.env, token_ids)
                return self.score_expr(hyp)
        except CandidateVerificationTimeout as e:
            return IVPRewardInfo(
                reward=-25.0, valid_parse=False, certified_ivp=False, elite_eligible=False,
                ode_rel_mse=1e6, ref_nrmse=1e6, anchor_rmse=1e6,
                expression=hyp, fitted_expression=None, fitted_coefficients={},
                error=f"TIMEOUT: {e}",
            )
        except BaseException as e:
            return IVPRewardInfo(
                reward=-25.0, valid_parse=False, certified_ivp=False, elite_eligible=False,
                ode_rel_mse=1e6, ref_nrmse=1e6, anchor_rmse=1e6,
                expression=hyp, fitted_expression=None, fitted_coefficients={},
                error=f"{type(e).__name__}: {e}",
            )

    def evaluate_generated(self, generated, gen_len, cache: Optional[dict] = None):
        candidates = generated_to_candidates(self.env, generated, gen_len)
        infos: List[IVPRewardInfo] = []
        hits = misses = 0
        for ids, _ in candidates:
            key = tuple(int(x) for x in ids)
            if cache is not None and key in cache:
                info = cache[key]
                hits += 1
            else:
                info = self.score_ids(ids)
                if cache is not None:
                    cache[key] = info
                misses += 1
            infos.append(info)
        return candidates, infos, hits, misses


def summarize_ivp(candidates, infos, top_k=5):
    order = sorted(range(len(infos)), key=lambda i: infos[i].reward, reverse=True)
    rows = []
    for i in order[:top_k]:
        ids, words = candidates[i]
        z = infos[i]
        rows.append({
            "idx": i,
            "reward": float(z.reward),
            "certified": bool(z.certified_ivp),
            "elite_eligible": bool(z.elite_eligible),
            "ode_rel": float(z.ode_rel_mse),
            "ref_nrmse": float(z.ref_nrmse),
            "anchor_rmse": float(z.anchor_rmse),
            "expr": str(z.expression) if z.expression is not None else "<parse error>",
            "fitted_expr": str(z.fitted_expression) if z.fitted_expression is not None else "<invalid>",
            "coeffs": dict(z.fitted_coefficients),
            "tokens": " ".join(words),
        })
    return rows
