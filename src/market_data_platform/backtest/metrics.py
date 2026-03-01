"""Portfolio performance metrics.

All functions accept a ``pd.Series`` of daily returns and return scalar
floats.  The ``compute_metrics`` convenience function runs all metrics at
once and returns a ``MetricsReport`` dataclass.

Functions
---------
sharpe_ratio        : annualized Sharpe ratio
sortino_ratio       : annualized Sortino ratio (downside deviation)
max_drawdown        : maximum peak-to-trough drawdown (positive fraction)
hit_rate            : fraction of days with positive returns
annualized_return   : geometric annualized return
turnover            : daily average absolute weight change
compute_metrics     : aggregate gross metrics into MetricsReport
compute_net_metrics : aggregate gross + net (cost-adjusted) metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    hit_rate: float
    turnover: Optional[float] = None
    # Phase 3 — cost-adjusted metrics (None when no cost model is applied)
    net_sharpe_ratio: Optional[float] = None
    net_annualized_return: Optional[float] = None
    total_cost_bps: Optional[float] = None
    avg_cost_per_trade_bps: Optional[float] = None
    cost_drag_annual_bps: Optional[float] = None
    breakeven_turnover: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "MetricsReport",
            f"  Annualized return : {self.annualized_return:+.2%}",
            f"  Sharpe ratio      : {self.sharpe_ratio:.3f}",
            f"  Sortino ratio     : {self.sortino_ratio:.3f}",
            f"  Max drawdown      : {self.max_drawdown:.2%}",
            f"  Hit rate          : {self.hit_rate:.2%}",
        ]
        if self.turnover is not None:
            lines.append(f"  Turnover (daily)  : {self.turnover:.4f}")
        if self.net_sharpe_ratio is not None:
            lines.extend(
                [
                    f"  Net Sharpe ratio  : {self.net_sharpe_ratio:.3f}",
                    f"  Net ann. return   : {self.net_annualized_return:+.2%}",
                    f"  Total cost (bps)  : {self.total_cost_bps:.1f}",
                    f"  Cost drag (bps/y) : {self.cost_drag_annual_bps:.1f}",
                    f"  Breakeven t/o     : {self.breakeven_turnover:.2f}",
                ]
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual metric functions                                                   #
# --------------------------------------------------------------------------- #


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio.

    Returns ``inf`` when std = 0 and mean > 0, ``nan`` when std = 0 and
    mean = 0, ``-inf`` when std = 0 and mean < 0.
    """
    std = returns.std()
    if std == 0:
        mean = returns.mean()
        if mean > 0:
            return float("inf")
        if mean < 0:
            return float("-inf")
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sortino ratio using downside deviation.

    Only negative returns contribute to the downside std.  Returns ``inf``
    when there are no negative returns (all positive), ``nan`` for an empty
    series.
    """
    if returns.empty:
        return float("nan")
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf")
    down_std = downside.std()
    if down_std == 0:
        return float("inf")
    return float(returns.mean() / down_std * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction.

    A return of 0.20 means the portfolio lost 20% from its peak at some
    point during the period.
    """
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown = (cumulative - rolling_peak) / rolling_peak
    return float(abs(drawdown.min()))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of trading days with a positive return."""
    if returns.empty:
        return float("nan")
    return float((returns > 0).sum() / len(returns))


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """Geometric annualized return.

    Computed as ``(product of (1 + r_i))^(periods / n) − 1``.
    """
    n = len(returns)
    if n == 0:
        return 0.0
    total_growth = (1 + returns).prod()
    return float(total_growth ** (periods / n) - 1)


def turnover(weights_df: pd.DataFrame) -> float:
    """Daily average absolute weight change (one-way turnover).

    Parameters
    ----------
    weights_df:
        DataFrame with dates as index and symbols as columns.  Each cell
        contains the portfolio weight for that symbol on that date.
    """
    # min_count=1 ensures the first row (all NaN diffs) is NaN rather than 0,
    # so it is excluded from the mean — matching standard industry convention.
    return float(weights_df.diff().abs().sum(axis=1, min_count=1).mean())


# --------------------------------------------------------------------------- #
# Convenience aggregator                                                        #
# --------------------------------------------------------------------------- #


def compute_metrics(
    returns: pd.Series,
    periods: int = 252,
    weights_df: Optional[pd.DataFrame] = None,
) -> MetricsReport:
    """Compute all metrics and return a ``MetricsReport``.

    Parameters
    ----------
    returns:
        Daily portfolio return series.
    periods:
        Number of trading periods per year (252 for daily returns).
    weights_df:
        Optional weights DataFrame for turnover computation.
    """
    report = MetricsReport(
        annualized_return=annualized_return(returns, periods),
        sharpe_ratio=sharpe_ratio(returns, periods),
        sortino_ratio=sortino_ratio(returns, periods),
        max_drawdown=max_drawdown(returns),
        hit_rate=hit_rate(returns),
        turnover=turnover(weights_df) if weights_df is not None else None,
    )
    logger.info(report.summary())
    return report


def compute_net_metrics(
    gross_returns: pd.Series,
    cost_report,  # CostReport (avoid circular import — accept duck-typed)
    periods: int = 252,
) -> MetricsReport:
    """Compute gross and net (cost-adjusted) metrics together.

    Parameters
    ----------
    gross_returns:
        Daily gross portfolio return series.
    cost_report:
        A ``CostReport`` instance with ``net_returns``, ``total_cost_bps``,
        and ``cost_per_trade`` attributes.
    periods:
        Trading periods per year.  Default 252.

    Returns
    -------
    MetricsReport
        Gross metrics plus Phase 3 cost fields populated.
    """
    net_returns = cost_report.net_returns

    gross_ann = annualized_return(gross_returns, periods)
    net_ann = annualized_return(net_returns, periods) if net_returns is not None else float("nan")
    drag_bps = (gross_ann - net_ann) * 10_000

    n_trades = len(cost_report.cost_per_trade)
    total_cost_bps = cost_report.total_cost_bps
    avg_cost_per_trade_bps = (total_cost_bps / n_trades) if n_trades > 0 else 0.0

    # Breakeven turnover: number of weight-unit trades per year that would
    # consume all gross alpha.  Formula: gross_alpha_bps / avg_cost_bps_per_trade.
    if avg_cost_per_trade_bps > 0:
        breakeven_to = (gross_ann * 10_000) / avg_cost_per_trade_bps
    else:
        breakeven_to = float("inf")

    report = MetricsReport(
        annualized_return=gross_ann,
        sharpe_ratio=sharpe_ratio(gross_returns, periods),
        sortino_ratio=sortino_ratio(gross_returns, periods),
        max_drawdown=max_drawdown(gross_returns),
        hit_rate=hit_rate(gross_returns),
        net_sharpe_ratio=sharpe_ratio(net_returns, periods) if net_returns is not None else float("nan"),
        net_annualized_return=net_ann,
        total_cost_bps=total_cost_bps,
        avg_cost_per_trade_bps=avg_cost_per_trade_bps,
        cost_drag_annual_bps=drag_bps,
        breakeven_turnover=breakeven_to,
    )
    logger.info(report.summary())
    return report
