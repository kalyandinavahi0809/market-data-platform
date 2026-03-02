# Market Data Platform

A hedge-fund-grade market data engineering and quantitative research infrastructure built to simulate real buy-side systems. Designed to demonstrate production-quality data engineering for Quantitative Data Engineer and Market Data Platform Engineer roles.

## Architecture

```
Vendor/Exchange (yfinance, Binance)
        │
        ▼
┌─────────────────────┐
│   Ingestion Layer   │  batch_ingest.py + ingest_universe.py
│   23 symbols        │  20 equities + 3 crypto, 6 years history
└────────┬────────────┘
         │ Hive-partitioned Parquet
         ▼
┌─────────────────────┐
│     Raw Layer       │  data/raw/symbol=X/date=Y/part-Y.parquet
│  Immutable store    │  Schema-validated via Pandera
└────────┬────────────┘
         │ Normalize + Deduplicate
         ▼
┌─────────────────────┐
│  Canonical Layer    │  data/canonical/symbol=X/date=Y/part-Y.parquet
│  Cleaned + typed    │  Log returns, VWAP, quality flags
└────────┬────────────┘
         │ DuckDB views (hive_partitioning=true)
         ▼
┌─────────────────────┐
│   Research Layer    │  Momentum, volatility, volume z-score
│   Feature engine    │  Cross-sectional ranking, forward returns
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Backtest Engine   │  Long/short quintile portfolio
│   + Cost Modeling   │  Slippage, commission, bid-ask spread
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Walk-Forward OOS   │  Rolling train/test windows
│  Regime Analysis    │  LOW / NORMAL / HIGH volatility regimes
└─────────────────────┘
```

## Phases

### Phase 1 — Storage & Canonical Layer

- **DuckDB client** with connection management, view registration over Hive-partitioned parquet, and full introspection API (`symbol_summary`, `row_count`, `describe`)
- **Schema registry** using Pandera with contracts for raw, canonical, and research layers
- **Canonical normalizer** — reads raw partitions, enforces schema, deduplicates on `(symbol, ts_utc)`, computes log returns, writes idempotent partitions
- **Quality checks** — OHLCV sanity (`high >= low`, `close` in `[low, high]`), gap detection (configurable for equity vs crypto calendars), freshness validation
- **Universe ingestion** — 23 symbols (20 US large-cap equities + BTC, ETH, SOL) from 2020-01-01 to present

### Phase 2 — Research Layer & Backtest Engine

- **Feature engine** — `log_return_1d/5d/20d`, `vol_20d` (annualized realized vol), `volume_zscore` (rolling 20d z-score). All vectorized, no iterrows.
- **Forward returns** — `fwd_return_1d/5d`, strictly point-in-time safe. Identity enforced: `fwd_return_1d[t] == log_return_1d[t+1]`
- **Cross-sectional ranking** — rank and z-score normalization across the universe at each timestamp. Handles missing symbols gracefully.
- **Backtest engine** — long top quintile, short bottom quintile by 20d momentum rank. Dollar-neutral, equal-weight, daily rebalance. Vectorized portfolio simulation.
- **Performance metrics** — Sharpe, Sortino, max drawdown, hit rate, annualized return, turnover

### Phase 3 — Transaction Cost Modeling

- **Slippage models** — `LinearSlippage`, `VolatilitySlippage`, `SquareRootImpact` (Almgren approximation, industry standard)
- **Commission models** — `FixedCommission`, `BpsCommission`, `TieredCommission`
- **Spread models** — `ConstantSpread`, `VolatilitySpread` (widens in stress)
- **CostEngine** — composable, applies all three cost components to a trades DataFrame and returns `CostReport` with full attribution (slippage / commission / spread breakdown)
- **Net metrics** — `compute_net_metrics()` produces gross vs net Sharpe, cost drag in bps, breakeven turnover

### Phase 4 — Walk-Forward Validation & Regime Analysis

- **WalkForwardSplitter** — zero-leakage rolling train/test splits. Configurable `train_period`, `test_period`, `step_size`, `min_train`.
- **OOSEvaluator** — runs backtest on each test window using only train-period signal parameters. Stitches test windows into a continuous OOS return series. Produces `OOSReport` with IS vs OOS Sharpe comparison and degradation %.
- **VolatilityRegimeFilter** — classifies each date as `LOW_VOL`, `NORMAL_VOL`, or `HIGH_VOL` using rolling realized vol percentiles estimated on training data only. Reports regime-conditional Sharpe ratios.

## Universe

| Sector | Symbols |
|---|---|
| Technology | AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO |
| Financials | JPM, BAC, V, MA |
| Healthcare | JNJ, UNH, MRK, PFE |
| Energy | XOM, CVX |
| Consumer | PG, HD |
| EV | TSLA |
| Crypto | BTC-USD, ETH-USD, SOL-USD |

Coverage: 2020-01-01 → present | ~38,000 daily bars total

## Signal Definition

```
Hypothesis:
  Cross-sectional momentum in equities and crypto persists
  over 20-day horizons due to investor underreaction.

Signal:
  r_{i,t}^{20d} = ln(close_{i,t} / close_{i,t-20})
  rank_{i,t}    = cross_sectional_rank(r^{20d}) normalized to [-1, 1]

Universe:    23 symbols (20 equity + 3 crypto)
Holding:     Daily rebalance
Long:        top quintile  (rank > 0.6)
Short:       bottom quintile (rank < -0.6)
Weighting:   Equal weight within each quintile
Costs:       5bps slippage + 5bps commission + 5bps half-spread
```

## Project Structure

```
market-data-platform/
├── config/
│   └── universe.yaml                  # 23-symbol universe definition
├── data/
│   ├── raw/                           # Hive-partitioned raw parquet
│   └── canonical/                     # Hive-partitioned canonical parquet
├── docs/
│   ├── architecture.md
│   └── data_model.md                  # Schema contracts for all layers
├── scripts/
│   └── ingest_universe.py             # Multi-symbol ingestion runner
├── src/market_data_platform/
│   ├── ingestion/
│   │   └── batch_ingest.py            # Single-symbol yfinance ingestion
│   ├── storage/
│   │   ├── duckdb_client.py           # Connection manager + query interface
│   │   └── schema_registry.py         # Pandera schema contracts
│   ├── canonical/
│   │   ├── normalizer.py              # Raw → canonical transform
│   │   └── writer.py                  # Idempotent partition writer
│   ├── quality/
│   │   └── checks.py                  # OHLCV sanity, gaps, freshness
│   ├── research/
│   │   ├── features.py                # Log returns, vol, volume z-score
│   │   ├── cross_section.py           # Rank/z-score normalization
│   │   └── forward_returns.py         # Point-in-time safe fwd returns
│   ├── backtest/
│   │   ├── engine.py                  # Vectorized L/S portfolio simulation
│   │   └── metrics.py                 # Sharpe, Sortino, drawdown, turnover
│   ├── costs/
│   │   ├── slippage.py                # Linear, Volatility, SquareRoot models
│   │   ├── commission.py              # Fixed, Bps, Tiered commission
│   │   ├── spread.py                  # Constant and Volatility spread
│   │   └── cost_engine.py             # CostEngine + CostReport
│   └── validation/
│       ├── walk_forward.py            # WalkForwardSplitter
│       ├── oos_evaluator.py           # OOSEvaluator + OOSReport
│       └── regime_filter.py           # VolatilityRegimeFilter
└── tests/                             # 190 tests across all layers
    ├── storage/
    ├── canonical/
    ├── quality/
    ├── research/
    ├── backtest/
    ├── costs/
    └── validation/
```

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/kalyandinavahi0809/market-data-platform.git
cd market-data-platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Ingest full universe (23 symbols, 2020 → present)
make ingest-universe

# 3. Canonicalize raw data
make canonicalize

# 4. Run quality checks
make quality-check

# 5. Run full test suite
pytest tests/ -v

# 6. Run walk-forward validation
make walk-forward
```

## Data Verification

```python
from market_data_platform.storage.duckdb_client import DuckDBClient

client = DuckDBClient()
client.refresh_views()

# Check all 23 symbols are present with correct row counts
print(client.symbol_summary("canonical_ohlcv"))

# Query specific symbol
df = client.query("""
    SELECT symbol, ts_utc, close, volume
    FROM canonical_ohlcv
    WHERE symbol = 'AAPL'
    ORDER BY ts_utc DESC
    LIMIT 5
""")
print(df)
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Storage format | Apache Parquet (Hive-partitioned) |
| Query engine | DuckDB |
| Schema validation | Pandera |
| Data source | yfinance |
| Testing | pytest (190 tests) |
| Version control | Git / GitHub |

## Engineering Standards

- **Layered architecture** — raw → canonical → features → signals → backtest
- **Idempotent writes** — re-running ingestion or canonicalization produces identical output
- **Schema contracts** — Pandera validation at every layer boundary
- **Point-in-time correctness** — no lookahead bias in features or forward returns
- **Vectorized computation** — no iterrows anywhere in the research or backtest stack
- **Composable cost modeling** — any combination of slippage, commission, and spread models
- **190 unit tests** — comprehensive coverage across all modules

## Key Design Decisions

**DuckDB over CSV** — Hive-partitioned parquet with DuckDB as the query engine gives columnar performance and SQL access without requiring a running database server. Partition pruning on symbol and date makes per-symbol queries fast even at scale.

**Pandera for schema validation** — Data contracts are defined in code, not documentation. Every layer boundary enforces types, nullability, and value constraints before data moves downstream.

**Square Root Impact model** — The Almgren approximation (impact = σ × √(ADV_fraction)) is the industry standard for equity market impact and is used by most real buy-side desks. Linear slippage is available for baseline comparison.

**Walk-forward over simple train/test split** — A single train/test split is insufficient for time-series strategy validation. Walk-forward testing with multiple rolling windows provides a more robust estimate of out-of-sample performance and exposes regime sensitivity.
