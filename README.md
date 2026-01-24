# market-data-platform

Institutional-style market data platform that supports:
- streaming + batch ingestion
- raw → canonical → research-ready layers
- replay/backfill, idempotency, and data quality checks

## Architecture (high level)
Source (vendor/exchange) → Kafka → Ingestion → Raw Store → Canonical Model → Research Layer

## Milestones
1) Ingest sample market data (batch) into raw layer
2) Add streaming ingestion + replay
3) Canonicalize schema + time-series modeling
4) Data quality + observability

## Tech
Python, SQL, Kafka, Docker

## Quickstart

To set up a virtual environment and run the batch ingestion:

```bash
python -m venv .venv
source .venv/bin/activate       # on Windows use .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m market_data_platform.ingestion.batch_ingest \
    --symbol BTC-USD \
    --start 2025-01-01 \
    --end 2025-01-05
```

The ingested Parquet files will appear under `data/raw/symbol=BTC-USD/` in date-partitioned folders (e.g., `date=2025-01-01/part-2025-01-01.parquet`).
