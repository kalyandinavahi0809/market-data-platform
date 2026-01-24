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
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # on Windows use .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# install the package in editable mode
pip install -e .

# run the ingestion script
python -m market_data_platform.ingestion.batch_ingest \
    --symbol BTC-USD \
    --start 2025-01-01 \
    --end 2025-01-05
