"""Ingest all symbols defined in config/universe.yaml.

Usage
-----
    python scripts/ingest_universe.py
    python scripts/ingest_universe.py --config config/universe.yaml
    python scripts/ingest_universe.py --start 2024-01-01 --end 2024-12-31

Each symbol is ingested independently. Failures are caught and logged so that
one bad symbol does not abort the full run. A summary table is printed at the
end showing pass/fail status per symbol.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "universe.yaml"


def load_universe(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_symbols(universe: dict) -> List[str]:
    symbols: List[str] = []
    for asset_class in ("equities", "crypto"):
        symbols.extend(universe.get(asset_class, []))
    return symbols


def ingest_symbol(
    symbol: str,
    start: str,
    end: str,
    out_dir: str,
    source: str,
) -> bool:
    """Call batch_ingest for a single symbol.  Returns True on success."""
    cmd = [
        sys.executable,
        "-m",
        "market_data_platform.ingestion.batch_ingest",
        "--symbol", symbol,
        "--start", start,
        "--end", end,
        "--out", out_dir,
        "--source", source,
    ]
    logger.info("Ingesting %s (%s → %s)", symbol, start, end)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("  OK   %s", symbol)
            return True
        else:
            logger.error(
                "  FAIL %s (exit %d)\n%s",
                symbol,
                result.returncode,
                result.stderr.strip(),
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error("  FAIL %s — timed out after 120s", symbol)
        return False
    except Exception as exc:
        logger.error("  FAIL %s — unexpected error: %s", symbol, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all universe symbols")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to universe.yaml (default: config/universe.yaml)",
    )
    parser.add_argument("--start", default=None, help="Override start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Override end date YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw", help="Raw data output root")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Universe config not found: %s", config_path)
        sys.exit(1)

    universe = load_universe(config_path)
    defaults = universe.get("defaults", {})

    start = args.start or defaults.get("start_date", "2020-01-01")
    end = args.end or defaults.get("end_date") or str(date.today())
    source = defaults.get("source", "yfinance")

    symbols = collect_symbols(universe)
    if not symbols:
        logger.error("No symbols found in %s", config_path)
        sys.exit(1)

    logger.info(
        "Starting universe ingest: %d symbols, %s → %s", len(symbols), start, end
    )

    results: Dict[str, bool] = {}
    for symbol in symbols:
        results[symbol] = ingest_symbol(symbol, start, end, args.out, source)

    passed = [s for s, ok in results.items() if ok]
    failed = [s for s, ok in results.items() if not ok]

    print("\n" + "=" * 55)
    print(f"Universe ingest complete: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("Failed symbols:")
        for s in failed:
            print(f"  - {s}")
    print("=" * 55)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
