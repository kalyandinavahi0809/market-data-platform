"""Out-of-sample (OOS) walk-forward evaluator.

Runs the full research → backtest pipeline on each walk-forward window and
stitches together a continuous out-of-sample return series.

Critical design constraint
--------------------------
No information from the test window is used during feature computation for
the test period.  Features that require *n* days of history (e.g. 20-day
vol) may use up to *n* days of lookback from the **training** window — this
is intentional and does **not** constitute lookahead bias.

Typical usage
-------------
>>> splitter = WalkForwardSplitter(train_period=504, test_period=63)
>>> evaluator = OOSEvaluator(splitter=splitter, cost_engine=engine)
>>> report = evaluator.run(canonical_df)
>>> print(evaluator.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from market_data_platform.backtest.engine import run_backtest, run_with_costs
from market_data_platform.backtest.metrics import (
    annualized_return,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from market_data_platform.research.cross_section import compute_cross_section
from market_data_platform.research.features import compute_features
from market_data_platform.research.forward_returns import add_forward_returns
from market_data_platform.validation.walk_forward import WalkForwardSplitter

logger = logging.getLogger(__name__)

# Max days of training lookback included when computing OOS features.
# Equal to the longest feature window (20-day log return / vol).
_FEATURE_LOOKBACK: int = 20


# --------------------------------------------------------------------------- #
# Result dataclasses                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowResult:
    """Metrics for a single walk-forward window.

    Attributes
    ----------
    window_idx:
        Zero-based window index.
    train_start / train_end:
        First and last training dates.
    test_start / test_end:
        First and last test (OOS) dates.
    is_sharpe:
        Annualised Sharpe ratio computed on the *training* data only.
    oos_sharpe:
        Annualised Sharpe ratio on the *test* data (gross of costs).
    oos_return:
        Geometric annualised return on the test data (gross).
    oos_max_drawdown:
        Maximum drawdown on the test data (gross).
    oos_hit_rate:
        Fraction of positive test-period return days (gross).
    oos_net_sharpe:
        OOS Sharpe after transaction costs (``nan`` if no cost engine).
    n_train_days:
        Number of unique dates in the training window.
    n_test_days:
        Number of unique dates in the test window.
    """

    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_sharpe: float
    oos_sharpe: float
    oos_return: float
    oos_max_drawdown: float
    oos_hit_rate: float
    oos_net_sharpe: float
    n_train_days: int
    n_test_days: int


@dataclass
class OOSReport:
    """Aggregated walk-forward validation results.

    Attributes
    ----------
    n_windows:
        Total number of walk-forward windows evaluated.
    oos_sharpe:
        Sharpe ratio computed on the **stitched** OOS return series.
    oos_sortino:
        Sortino ratio on the stitched OOS series.
    oos_max_drawdown:
        Max drawdown on the stitched OOS series.
    oos_hit_rate:
        Hit rate on the stitched OOS series.
    oos_annualized_return:
        Geometric annualised return on the stitched OOS series.
    oos_net_sharpe:
        OOS Sharpe after costs (``nan`` if no cost engine).
    is_sharpe:
        Average in-sample Sharpe across all windows.
    sharpe_degradation_pct:
        ``(is_sharpe − oos_sharpe) / |is_sharpe| × 100``.
        Positive = strategy degrades OOS (typical); negative = OOS better.
        ``nan`` when ``is_sharpe ≈ 0``.
    window_results:
        Tuple of per-window :class:`WindowResult` objects.
    oos_returns:
        Stitched daily OOS portfolio return series (gross).
    """

    n_windows: int
    oos_sharpe: float
    oos_sortino: float
    oos_max_drawdown: float
    oos_hit_rate: float
    oos_annualized_return: float
    oos_net_sharpe: float
    is_sharpe: float
    sharpe_degradation_pct: float
    window_results: Tuple[WindowResult, ...]
    oos_returns: pd.Series


# --------------------------------------------------------------------------- #
# Evaluator                                                                    #
# --------------------------------------------------------------------------- #


class OOSEvaluator:
    """Runs walk-forward validation across all windows.

    For each window the evaluator:

    1. Computes in-sample features → cross-section → backtest on *train* data.
    2. Prepends the last :data:`_FEATURE_LOOKBACK` training days to the test
       window before running ``compute_features`` (prevents NaN at test start
       without leaking test information).
    3. Extracts only test-period rows from the feature-enriched frame.
    4. Runs the backtest (and optionally the cost engine) on test-only data.
    5. Records per-window IS/OOS metrics.

    The final ``OOSReport`` stitches all test-period returns into a single
    continuous OOS series, then computes aggregate statistics.

    Parameters
    ----------
    splitter:
        A configured :class:`~validation.walk_forward.WalkForwardSplitter`.
    cost_engine:
        Optional :class:`~costs.cost_engine.CostEngine`.  When provided,
        ``oos_net_sharpe`` is populated in window results and the report.
    """

    def __init__(
        self,
        splitter: WalkForwardSplitter,
        cost_engine=None,
    ) -> None:
        self.splitter = splitter
        self.cost_engine = cost_engine
        self._report: Optional[OOSReport] = None

    # ---------------------------------------------------------------------- #
    # Main entry point                                                         #
    # ---------------------------------------------------------------------- #

    def run(self, canonical_df: pd.DataFrame) -> OOSReport:
        """Execute walk-forward validation on *canonical_df*.

        Parameters
        ----------
        canonical_df:
            Full canonical OHLCV DataFrame for all symbols.  Must contain
            ``ts_utc``, ``symbol``, ``open``, ``high``, ``low``, ``close``,
            ``volume`` columns.

        Returns
        -------
        OOSReport
            Contains all window results and the stitched OOS return series.

        Raises
        ------
        ValueError
            If the splitter produces zero windows.
        """
        splits = self.splitter.split(canonical_df)

        if not splits:
            raise ValueError(
                "WalkForwardSplitter produced zero windows. "
                "Increase data length or reduce train/test periods."
            )

        window_results = []
        all_oos_gross: list[pd.Series] = []
        all_oos_net: list[pd.Series] = []

        for i, (train_df, test_df) in enumerate(splits):
            logger.info("Walk-forward window %d / %d …", i + 1, len(splits))

            wr, oos_gross, oos_net = self._eval_window(i, train_df, test_df)
            window_results.append(wr)
            all_oos_gross.append(oos_gross)
            if oos_net is not None:
                all_oos_net.append(oos_net)

        # ------------------------------------------------------------------ #
        # Stitch OOS returns                                                  #
        # ------------------------------------------------------------------ #
        stitched = (
            pd.concat(all_oos_gross)
            .sort_index()
        )
        stitched.name = "oos_portfolio_return"
        clean = stitched.dropna()

        # ------------------------------------------------------------------ #
        # Aggregate OOS metrics                                               #
        # ------------------------------------------------------------------ #
        oos_sh = sharpe_ratio(clean) if len(clean) > 1 else float("nan")
        oos_so = sortino_ratio(clean) if len(clean) > 1 else float("nan")
        oos_dd = max_drawdown(clean)
        oos_hr = hit_rate(clean) if len(clean) > 0 else float("nan")
        oos_ar = annualized_return(clean)

        if all_oos_net:
            net_clean = pd.concat(all_oos_net).sort_index().dropna()
            oos_net_sh = sharpe_ratio(net_clean) if len(net_clean) > 1 else float("nan")
        else:
            oos_net_sh = float("nan")

        valid_is = [w.is_sharpe for w in window_results if not np.isnan(w.is_sharpe)]
        avg_is_sh = float(np.mean(valid_is)) if valid_is else float("nan")

        if abs(avg_is_sh) > 1e-10 and not np.isnan(avg_is_sh) and not np.isnan(oos_sh):
            degradation = (avg_is_sh - oos_sh) / abs(avg_is_sh) * 100.0
        else:
            degradation = float("nan")

        report = OOSReport(
            n_windows=len(window_results),
            oos_sharpe=oos_sh,
            oos_sortino=oos_so,
            oos_max_drawdown=oos_dd,
            oos_hit_rate=oos_hr,
            oos_annualized_return=oos_ar,
            oos_net_sharpe=oos_net_sh,
            is_sharpe=avg_is_sh,
            sharpe_degradation_pct=degradation,
            window_results=tuple(window_results),
            oos_returns=stitched,
        )

        self._report = report
        logger.info(
            "Walk-forward complete: %d windows, OOS Sharpe=%.3f, IS Sharpe=%.3f",
            report.n_windows,
            report.oos_sharpe,
            report.is_sharpe,
        )
        return report

    def summary(self) -> str:
        """Return a formatted text summary of the most recent OOS report.

        Returns
        -------
        str
            Multi-line performance table.  Returns an informative placeholder
            string if :meth:`run` has not been called.
        """
        if self._report is None:
            return "OOSEvaluator: not yet run — call run(canonical_df) first."

        r = self._report

        def _fmt(v: float, fmt: str = ".3f") -> str:
            return f"{v:{fmt}}" if not (np.isnan(v) or np.isinf(v)) else "n/a"

        lines = [
            "OOS Validation Summary",
            "======================",
            f"  Windows              : {r.n_windows}",
            f"  IS Sharpe (avg)      : {_fmt(r.is_sharpe)}",
            f"  OOS Sharpe           : {_fmt(r.oos_sharpe)}",
            f"  Sharpe degradation   : {_fmt(r.sharpe_degradation_pct, '.1f')} %",
            f"  OOS Annualized rtn   : {_fmt(r.oos_annualized_return * 100, '.2f')} %",
            f"  OOS Sortino          : {_fmt(r.oos_sortino)}",
            f"  OOS Max Drawdown     : {_fmt(r.oos_max_drawdown * 100, '.2f')} %",
            f"  OOS Hit Rate         : {_fmt(r.oos_hit_rate * 100, '.2f')} %",
            f"  OOS Net Sharpe       : {_fmt(r.oos_net_sharpe)}",
        ]
        return "\n".join(lines)

    # ---------------------------------------------------------------------- #
    # Per-window evaluation                                                    #
    # ---------------------------------------------------------------------- #

    def _eval_window(
        self,
        idx: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """Evaluate one walk-forward window.

        Returns
        -------
        (WindowResult, gross_oos_returns, net_oos_returns | None)
        """
        train_dates = sorted(train_df["ts_utc"].unique())
        test_dates = sorted(test_df["ts_utc"].unique())

        # ------------------------------------------------------------------ #
        # In-sample: full pipeline on train data                             #
        # ------------------------------------------------------------------ #
        train_feat = compute_features(train_df)
        train_cs = compute_cross_section(train_feat)
        train_full = add_forward_returns(train_cs)
        is_ret = run_backtest(train_full).dropna()
        is_sh = sharpe_ratio(is_ret) if len(is_ret) > 1 else float("nan")

        # ------------------------------------------------------------------ #
        # OOS: prepend lookback from training to allow proper feature warmup  #
        # ------------------------------------------------------------------ #
        lookback_dates = set(train_dates[-_FEATURE_LOOKBACK:])
        lookback_df = train_df[train_df["ts_utc"].isin(lookback_dates)]

        combined = pd.concat([lookback_df, test_df], ignore_index=True)
        test_feat = compute_features(combined)
        test_cs = compute_cross_section(test_feat)
        test_full = add_forward_returns(test_cs)

        # Keep only rows that belong to the test window
        test_dates_set = set(test_dates)
        test_only = test_full[test_full["ts_utc"].isin(test_dates_set)].copy()

        # ------------------------------------------------------------------ #
        # OOS gross returns (and optionally net via cost engine)             #
        # ------------------------------------------------------------------ #
        if self.cost_engine is not None:
            cost_rep = run_with_costs(test_only, self.cost_engine)
            oos_gross = cost_rep.gross_returns
            oos_net: Optional[pd.Series] = cost_rep.net_returns
        else:
            oos_gross = run_backtest(test_only)
            oos_net = None

        oos_clean = oos_gross.dropna() if oos_gross is not None else pd.Series(dtype=float)

        oos_sh = sharpe_ratio(oos_clean) if len(oos_clean) > 1 else float("nan")
        oos_ar = annualized_return(oos_clean) if len(oos_clean) > 0 else float("nan")
        oos_dd = max_drawdown(oos_clean) if len(oos_clean) > 0 else float("nan")
        oos_hr = hit_rate(oos_clean) if len(oos_clean) > 0 else float("nan")

        if oos_net is not None:
            net_clean = oos_net.dropna()
            oos_net_sh = sharpe_ratio(net_clean) if len(net_clean) > 1 else float("nan")
        else:
            oos_net_sh = float("nan")

        wr = WindowResult(
            window_idx=idx,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            is_sharpe=is_sh,
            oos_sharpe=oos_sh,
            oos_return=oos_ar,
            oos_max_drawdown=oos_dd,
            oos_hit_rate=oos_hr,
            oos_net_sharpe=oos_net_sh,
            n_train_days=len(train_dates),
            n_test_days=len(test_dates),
        )

        return wr, oos_gross, oos_net
