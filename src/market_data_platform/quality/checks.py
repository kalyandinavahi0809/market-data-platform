"""Data quality checks for canonical OHLCV data.

Each check returns a CheckResult.  run_all_checks() aggregates them into a
QualityReport.

Checks
------
check_ohlcv_sanity     high >= low, close in [low, high], volume >= 0, no NULLs
check_gaps             missing business days between first and last date per symbol
check_freshness        latest partition age vs a staleness threshold (in days)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    detail: Optional[pd.DataFrame] = field(default=None, repr=False)


@dataclass
class QualityReport:
    symbol: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"QualityReport [{status}] symbol={self.symbol}"]
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  {mark} {c.name}: {c.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_ohlcv_sanity(df: pd.DataFrame) -> CheckResult:
    """Verify OHLCV price relationships and non-negativity constraints.

    Rules
    -----
    - high >= low
    - low <= close <= high
    - volume >= 0
    - No NULLs in price or volume columns
    """
    issues = []

    price_cols = ["open", "high", "low", "close", "volume"]
    null_counts = df[price_cols].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        issues.append(f"NULLs in {null_cols.to_dict()}")

    bad_hl = df[df["high"] < df["low"]]
    if not bad_hl.empty:
        issues.append(f"{len(bad_hl)} row(s) where high < low")

    bad_close_low = df[df["close"] < df["low"]]
    if not bad_close_low.empty:
        issues.append(f"{len(bad_close_low)} row(s) where close < low")

    bad_close_high = df[df["close"] > df["high"]]
    if not bad_close_high.empty:
        issues.append(f"{len(bad_close_high)} row(s) where close > high")

    bad_volume = df[df["volume"] < 0]
    if not bad_volume.empty:
        issues.append(f"{len(bad_volume)} row(s) with negative volume")

    if issues:
        return CheckResult(
            name="ohlcv_sanity",
            passed=False,
            message="; ".join(issues),
        )
    return CheckResult(
        name="ohlcv_sanity",
        passed=True,
        message=f"All {len(df)} rows pass OHLCV sanity",
    )


def check_gaps(
    df: pd.DataFrame,
    freq: str = "B",
) -> CheckResult:
    """Detect missing periods between the first and last date in *df*.

    Parameters
    ----------
    df:
        DataFrame with a ``ts_utc`` column for a single symbol.
    freq:
        Pandas date offset — ``"B"`` (business days) for equities, ``"D"``
        (calendar days) for crypto.

    Returns
    -------
    CheckResult
        Passes when no gaps are found.
    """
    if df.empty:
        return CheckResult(name="gaps", passed=True, message="No data to check")

    dates = pd.to_datetime(df["ts_utc"]).dt.normalize().dt.tz_localize(None)
    observed = set(dates)
    first, last = dates.min(), dates.max()
    expected = set(pd.date_range(start=first, end=last, freq=freq))
    missing = sorted(expected - observed)

    if missing:
        missing_strs = [d.strftime("%Y-%m-%d") for d in missing[:10]]
        suffix = f" (showing first 10)" if len(missing) > 10 else ""
        return CheckResult(
            name="gaps",
            passed=False,
            message=(
                f"{len(missing)} missing period(s){suffix}: "
                + ", ".join(missing_strs)
            ),
            detail=pd.DataFrame({"missing_date": missing}),
        )

    return CheckResult(
        name="gaps",
        passed=True,
        message=f"No gaps between {first.date()} and {last.date()}",
    )


def check_freshness(
    df: pd.DataFrame,
    max_staleness_days: int = 3,
    reference_date: Optional[date] = None,
) -> CheckResult:
    """Check that the most recent record is not older than *max_staleness_days*.

    Parameters
    ----------
    df:
        DataFrame with a ``ts_utc`` column.
    max_staleness_days:
        Maximum number of calendar days the latest record may lag behind
        *reference_date*.
    reference_date:
        The date to compare against.  Defaults to today.
    """
    if df.empty:
        return CheckResult(name="freshness", passed=False, message="No data")

    ref = reference_date or date.today()
    latest = pd.to_datetime(df["ts_utc"]).max().date()
    lag = (ref - latest).days

    if lag > max_staleness_days:
        return CheckResult(
            name="freshness",
            passed=False,
            message=(
                f"Latest record is {latest} — {lag} day(s) behind "
                f"{ref} (threshold: {max_staleness_days})"
            ),
        )

    return CheckResult(
        name="freshness",
        passed=True,
        message=f"Latest record is {latest} ({lag} day(s) old)",
    )


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def run_all_checks(
    df: pd.DataFrame,
    symbol: str,
    gap_freq: str = "B",
    max_staleness_days: int = 3,
    reference_date: Optional[date] = None,
) -> QualityReport:
    """Run all quality checks for *symbol* and return a QualityReport.

    Parameters
    ----------
    df:
        Canonical OHLCV DataFrame filtered to a single symbol.
    symbol:
        Symbol name (used for labelling the report).
    gap_freq:
        ``"B"`` for equities (business days), ``"D"`` for crypto.
    max_staleness_days:
        Freshness threshold in calendar days.
    reference_date:
        Override today for freshness check (useful in tests).
    """
    report = QualityReport(symbol=symbol)
    report.checks.append(check_ohlcv_sanity(df))
    report.checks.append(check_gaps(df, freq=gap_freq))
    report.checks.append(
        check_freshness(
            df,
            max_staleness_days=max_staleness_days,
            reference_date=reference_date,
        )
    )
    logger.info(report.summary())
    return report


def run_universe_checks(
    df: pd.DataFrame,
    crypto_symbols: Optional[List[str]] = None,
    max_staleness_days: int = 3,
    reference_date: Optional[date] = None,
) -> Dict[str, QualityReport]:
    """Run quality checks for every symbol in *df*.

    Parameters
    ----------
    df:
        Full canonical OHLCV DataFrame (all symbols).
    crypto_symbols:
        Symbols that trade every day — gaps checked with ``freq="D"``.
        Equity symbols use ``freq="B"`` (business days).
    max_staleness_days:
        Freshness threshold in calendar days.
    reference_date:
        Override today for freshness check.
    """
    crypto_symbols = set(crypto_symbols or [])
    reports: Dict[str, QualityReport] = {}
    for symbol, group in df.groupby("symbol"):
        freq = "D" if symbol in crypto_symbols else "B"
        reports[symbol] = run_all_checks(
            group,
            symbol=symbol,
            gap_freq=freq,
            max_staleness_days=max_staleness_days,
            reference_date=reference_date,
        )
    return reports
