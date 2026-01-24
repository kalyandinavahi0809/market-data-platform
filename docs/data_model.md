# Raw Layer Schema

The raw layer stores market data with the following schema. Canonical and research-layer schemas will be documented separately.

| Column          | Type      | Description                        |
| --------------- | --------- | ---------------------------------- |
| ts_utc          | timestamp | UTC timestamp of the record        |
| symbol          | string    | Instrument symbol (e.g., BTC-USD)  |
| open            | float     | Opening price                      |
| high            | float     | Highest price                      |
| low             | float     | Lowest price                       |
| close           | float     | Closing price                      |
| volume          | float     | Trade volume                       |
| ingested_at_utc | timestamp | When the record was ingested (UTC) |
| source          | string    | Data source (e.g., yfinance, mock) |

**Partition scheme:** Files are written to `data/raw/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<YYYY-MM-DD>.parquet`.
