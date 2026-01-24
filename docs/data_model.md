# Data Model

This document defines the canonical and research data models used in the market-data-platform. It outlines schemas for the raw, canonical, and research-ready tables, along with conventions for handling time-series market data.


# Raw Layer Schema

| Column          | Type      | Description                                  |
|-----------------|-----------|----------------------------------------------|
| ts_utc          | timestamp | UTC timestamp of the record                  |
| symbol          | string    | Instrument symbol (e.g., BTC‑USD)            |
| open            | float     | Opening price                                |
| high            | float     | Highest price                                |
| low             | float     | Lowest price                                 |
| close           | float     | Closing price                                |
| volume          | float     | Trade volume                                 |
| ingested_at_utc | timestamp | When the record was ingested (UTC)           |
| source          | string    | Data source (e.g., yfinance, mock)           |

**Partitioning scheme:** Records are written to folders structured as  
`data/raw/symbol=<SYMBOL>/date=<YYYY‑MM‑DD>/part-<YYYY‑MM‑DD>.parquet`.

**Example:** Ingesting BTC‑USD data for 2025‑01‑01 writes a file to  
`data/raw/symbol=BTC-USD/date=2025-01-01/part-2025-01-01.parquet`.
