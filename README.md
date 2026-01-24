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
