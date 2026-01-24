# Architecture

This document provides a detailed overview of the architecture for the **market-data-platform**, a hedge-fund style system for ingesting, storing, and serving market data to researchers and trading systems.

## High-Level Flow

1. **Source Feeds** – Exchange or vendor feeds (equities, crypto, macro) deliver raw tick or bar data via streaming APIs and batch files.  
2. **Ingestion Layer** – Event-driven ingestion services consume streaming feeds via Kafka and schedule batch jobs for historical backfills. Each record is enriched with metadata (timestamps, source ids) and written to the raw layer.  
3. **Raw Layer** – An immutable, append-only store that preserves every tick as received. Data is partitioned by date and instrument and stored in object storage.  
4. **Canonical Layer** – A normalized representation of market data with standardized schemas (e.g., order book snapshots, trades, quotes). This layer ensures schema evolution and handles deduplication, late data, and idempotent writes.  
5. **Research Layer** – A curated set of feature-ready tables (e.g., daily OHLCV, rolling volatility, momentum factors) designed for quant research and backtesting. Each table is versioned and point-in-time correct to avoid look-ahead bias.  
6. **Data Quality & Observability** – Cross-cutting services monitor volumes, freshness, nulls, and anomalies. SLA dashboards alert on ingestion delays or schema drift. Replay and backfill tooling supports recovery from failures.

## Components

- **Streaming Ingestion**: Python/Asyncio consumers connected to Kafka topics for real-time tick and quote data.  
- **Batch Ingestion**: Scheduled jobs (e.g., cron or Airflow) to pull historical files and reprocess them for backfilling.  
- **Kafka/Event Bus**: A distributed log for buffering inbound data and enabling decoupled consumers.  
- **Storage**: Cloud object storage (S3) for raw and canonical layers, and columnar analytics engine (DuckDB, Snowflake, or Postgres) for research queries.  
- **Transformations**: Idempotent ETL/ELT jobs that map raw records to canonical schemas and compute derived factors.  
- **API/Access**: SQL interfaces and Python libraries to query research tables and serve data to quants.

## Design Principles

- **Event-Driven**: Support high-frequency streaming data with low latency and exactly-once processing.  
- **Schema Evolution**: Use versioned schemas to accommodate changes in vendor feed formats without breaking consumers.  
- **Replayable Pipelines**: Enable reprocessing of historical data for backtesting and error recovery.  
- **Point-in-Time Correctness**: Provide data views that reflect what was known at any point to avoid look-ahead bias.  
- **Observability**: Build robust monitoring for data quality, freshness, and drift to ensure trustworthiness.

This architecture aims to mirror institutional hedge fund data platforms and showcase production-grade engineering practices.
