.PHONY: install lint test ingest-universe canonicalize quality-check features backtest

install:
	pip install -e .

lint:
	black src tests scripts

test:
	pytest tests/ -v

# Ingest all symbols defined in config/universe.yaml into data/raw/
ingest-universe:
	python scripts/ingest_universe.py

# Ingest a date range override: make ingest-universe START=2024-01-01 END=2024-12-31
ingest-universe-range:
	python scripts/ingest_universe.py --start $(START) --end $(END)

# Normalize raw → canonical for all symbols
canonicalize:
	python -c "\
from pathlib import Path; \
from market_data_platform.canonical.normalizer import normalize; \
from market_data_platform.canonical.writer import write_canonical; \
df = normalize(Path('data/raw')); \
n = write_canonical(df, Path('data/canonical')); \
print(f'Wrote {n} canonical partition(s)')"

# Compute research features from the canonical layer
features:
	python -c "\
from pathlib import Path; \
import pandas as pd; \
from market_data_platform.research.features import compute_features; \
files = sorted(Path('data/canonical').rglob('*.parquet')); \
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True); \
result = compute_features(df); \
print(f'Features computed: {len(result)} rows, {result[\"symbol\"].nunique()} symbols'); \
print(result[['symbol','ts_utc','log_return_1d','log_return_20d','vol_20d','volume_zscore']].tail())"

# Run the cross-sectional momentum backtest and print performance metrics
backtest:
	python -c "\
from pathlib import Path; \
import pandas as pd; \
from market_data_platform.research.features import compute_features; \
from market_data_platform.research.cross_section import compute_cross_section; \
from market_data_platform.research.forward_returns import add_forward_returns; \
from market_data_platform.backtest.engine import run_backtest; \
from market_data_platform.backtest.metrics import compute_metrics; \
files = sorted(Path('data/canonical').rglob('*.parquet')); \
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True); \
df = compute_features(df); \
df = compute_cross_section(df); \
df = add_forward_returns(df); \
returns = run_backtest(df); \
report = compute_metrics(returns.dropna()); \
print(report.summary())"

# Run data quality checks against the canonical layer
quality-check:
	python -c "\
from pathlib import Path; \
import pandas as pd; \
from market_data_platform.quality.checks import run_universe_checks; \
files = sorted(Path('data/canonical').rglob('*.parquet')); \
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True); \
crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD']; \
reports = run_universe_checks(df, crypto_symbols=crypto); \
failed = [s for s, r in reports.items() if not r.passed]; \
print(f'Quality check: {len(reports)} symbols, {len(failed)} failed'); \
[print(r.summary()) for r in reports.values() if not r.passed]"
