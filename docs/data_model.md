# Data Model

This document defines the schema contracts for each layer of the market-data-platform pipeline.

---

## Raw Layer

Immutable, append-only store written by the ingestion layer.
Partitioning: `data/raw/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<YYYY-MM-DD>.parquet`

| Column          | Type                | Nullable | Description                           |
|-----------------|---------------------|----------|---------------------------------------|
| ts_utc          | datetime64[us, UTC] | No       | UTC bar timestamp                     |
| symbol          | string              | No       | Instrument ticker (yfinance format)   |
| open            | float64             | No       | Opening price                         |
| high            | float64             | No       | Highest price                         |
| low             | float64             | No       | Lowest price                          |
| close           | float64             | No       | Closing price                         |
| volume          | float64             | No       | Trade volume                          |
| ingested_at_utc | datetime64[us, UTC] | No       | Wall-clock time of ingestion          |
| source          | string              | No       | Data source (e.g. `yfinance`, `mock`) |

**Example:** Ingesting BTC-USD data for 2025-01-01 writes to
`data/raw/symbol=BTC-USD/date=2025-01-01/part-2025-01-01.parquet`

---

## Canonical Layer

Validated, deduplicated OHLCV produced by `canonical/normalizer.py` and written by `canonical/writer.py`.
Duplicates on `(ts_utc, symbol)` are dropped; all price dtypes are enforced before write.
Partitioning: `data/canonical/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<YYYY-MM-DD>.parquet`

| Column | Type                | Nullable | Description       |
|--------|---------------------|----------|-------------------|
| ts_utc | datetime64[us, UTC] | No       | UTC bar timestamp |
| symbol | string              | No       | Instrument ticker |
| open   | float64             | No       | Opening price     |
| high   | float64             | No       | Highest price     |
| low    | float64             | No       | Lowest price      |
| close  | float64             | No       | Closing price     |
| volume | float64             | No       | Trade volume      |

**Quality invariants enforced before write:**
- `high >= low`
- `low <= close <= high`
- `volume >= 0`
- No NULLs in any column

---

## Research Features Layer

Feature-engineered columns added by `research/features.py` on top of the canonical OHLCV table.
All features are strictly backward-looking — no lookahead bias.

| Column          | Type                | Nullable | Description                                        |
|-----------------|---------------------|----------|----------------------------------------------------|
| ts_utc          | datetime64[us, UTC] | No       | UTC bar timestamp                                  |
| symbol          | string              | No       | Instrument ticker                                  |
| open            | float64             | No       | Opening price (from canonical)                     |
| high            | float64             | No       | Highest price (from canonical)                     |
| low             | float64             | No       | Lowest price (from canonical)                      |
| close           | float64             | No       | Closing price (from canonical)                     |
| volume          | float64             | No       | Trade volume (from canonical)                      |
| log_return_1d   | float64             | Yes      | 1-day log return: `ln(close[t] / close[t-1])`     |
| log_return_5d   | float64             | Yes      | 5-day log return: `ln(close[t] / close[t-5])`     |
| log_return_20d  | float64             | Yes      | 20-day log return: `ln(close[t] / close[t-20])`   |
| vol_20d         | float64             | Yes      | 20-day annualized volatility of `log_return_1d`   |
| volume_zscore   | float64             | Yes      | Volume z-score relative to 20-day rolling window  |

**Lookahead prevention:** sort by `(symbol, ts_utc)` before computing; all windows are backward-looking.
First `n` rows per symbol are NaN for the corresponding n-period features (min_periods enforced).

---

## Cross-Sectional Layer

Columns added by `research/cross_section.py`. Ranks are computed per `ts_utc` across all symbols
that have a valid (non-NaN) `log_return_20d` signal.

| Column     | Type    | Nullable | Description                                                |
|------------|---------|----------|------------------------------------------------------------|
| cs_rank    | float64 | Yes      | Cross-sectional rank normalized to [-1, +1]                |
| cs_zscore  | float64 | Yes      | Cross-sectional z-score of `log_return_20d`                |

**Normalization:** `cs_rank = (rank_0indexed / (n - 1)) × 2 − 1` where `n` = number of valid symbols on that date.
**Edge cases:**
- Single valid symbol on a date → `cs_rank = 0.0` (neutral)
- Symbol with NaN signal → `cs_rank = NaN`

---

## Forward Returns Layer

Columns added by `research/forward_returns.py`. Must be called **after** `compute_features` to prevent
lookahead contamination of feature columns.

| Column         | Type    | Nullable | Description                                                   |
|----------------|---------|----------|---------------------------------------------------------------|
| fwd_return_1d  | float64 | Yes      | 1-day forward log return: `ln(close[t+1] / close[t])`        |
| fwd_return_5d  | float64 | Yes      | 5-day forward log return: `ln(close[t+5] / close[t])`        |

**Identity:** `fwd_return_1d[t] == log_return_1d[t+1]` (exact equality).
**NaN at tail:** last 1 row has NaN `fwd_return_1d`; last 5 rows have NaN `fwd_return_5d`.

---

## Backtest Output

Produced by `backtest/engine.run_backtest()`. A daily `pd.Series` indexed by `ts_utc`.

| Field            | Type    | Description                                           |
|------------------|---------|-------------------------------------------------------|
| `portfolio_return` | float64 | Daily dollar-neutral portfolio return                |

**Portfolio construction:**
- Long: symbols with `cs_rank > 0.6` (top quintile), equal-weighted, gross = +1
- Short: symbols with `cs_rank < −0.6` (bottom quintile), equal-weighted, gross = −1
- Net exposure = 0 (dollar neutral)

**Performance metrics** (`backtest/metrics.MetricsReport`):

| Field              | Type            | Description                                            |
|--------------------|-----------------|--------------------------------------------------------|
| `annualized_return`| float           | Geometric annualized return                            |
| `sharpe_ratio`     | float           | Annualized Sharpe (mean / std × √252)                  |
| `sortino_ratio`    | float           | Annualized Sortino (mean / downside_std × √252)        |
| `max_drawdown`     | float           | Maximum peak-to-trough drawdown (positive value, 0–1) |
| `hit_rate`         | float           | Fraction of days with positive portfolio return        |
| `turnover`         | float or None   | Mean daily sum of absolute weight changes              |

---

## Transaction Cost Models (Phase 3)

Implemented in `costs/`. All models share the interface:
```python
apply(trade_size, price, vol, adv) -> float | array
```
All inputs accept both scalars and `pandas.Series` / `numpy` arrays.

### Slippage models (`costs/slippage.py`)

| Class | Formula | Default params |
|---|---|---|
| `LinearSlippage` | `notional × slippage_bps / 10_000` | `slippage_bps=5` |
| `VolatilitySlippage` | `notional × k × vol × sqrt(trade/adv)` | `k=0.1` |
| `SquareRootImpact` | `notional × sigma_coeff × vol × sqrt(trade/adv)` | `sigma_coeff=0.1` |

### Commission models (`costs/commission.py`)

| Class | Formula | Default params |
|---|---|---|
| `FixedCommission` | `per_trade` (flat) | `per_trade=1.0` |
| `BpsCommission` | `notional × bps / 10_000` | `bps=5` |
| `TieredCommission` | Lowest eligible bps rate | user-defined tiers |

**TieredCommission logic:** for each tier `(threshold, bps_rate)`, apply the lowest rate for which `notional >= threshold`.  Falls back to the highest rate when notional is below all thresholds.

### Spread models (`costs/spread.py`)

| Class | Formula | Default params |
|---|---|---|
| `ConstantSpread` | `notional × half_spread_bps / 10_000` | `half_spread_bps=5` |
| `VolatilitySpread` | `notional × max(min_bps, k × vol × 10_000) / 10_000` | `k=0.5, min_bps=2.0` |

**Half-spread convention:** one transaction pays half the bid-ask spread; the full round-trip cost is `2 × half_spread_bps`.

### CostEngine (`costs/cost_engine.py`)

Wires slippage + commission + spread into a single composable object.

```python
engine = CostEngine(LinearSlippage(5.0), BpsCommission(5.0), ConstantSpread(5.0))
report = engine.apply(trades_df, prices_df, gross_returns=gross_series)
```

**`trades_df` schema:**

| Column | Type | Description |
|---|---|---|
| symbol | string | Instrument ticker |
| date | datetime64[UTC] | Trade date |
| shares | float | Absolute share quantity |
| direction | int | +1 (long) or -1 (short) |

**`CostReport` fields:**

| Field | Type | Description |
|---|---|---|
| `cost_per_trade` | DataFrame | Per-trade breakdown (notional, slippage, commission, spread, total) |
| `total_cost_dollars` | float | Sum of all per-trade costs |
| `total_cost_bps` | float | `total_cost / total_notional × 10_000` |
| `gross_returns` | Series | Input gross returns (pass-through) |
| `net_returns` | Series | `gross_returns − daily_cost_drag` |
| `cost_attribution` | dict | `{"slippage": $, "commission": $, "spread": $}` |

**Cost assumptions:**
- Costs are applied once per trade (one-way, not round-trip)
- Spread is the half-spread (entry cost only; exit is a separate trade)
- `adv=0` in impact models → zero impact (safe handling)
- Missing symbols in `prices_df` → zero cost (graceful degradation)

---

## DuckDB Views

`DuckDBClient` registers both hive-partitioned stores as SQL views on connect:

| View name         | Underlying store   |
|-------------------|--------------------|
| `raw_ohlcv`       | `data/raw/`        |
| `canonical_ohlcv` | `data/canonical/`  |

Example query:
```sql
SELECT symbol,
       COUNT(*)          AS rows,
       MIN(ts_utc)::DATE AS first,
       MAX(ts_utc)::DATE AS last
FROM canonical_ohlcv
GROUP BY symbol
ORDER BY symbol;
```
