CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    ts_ms     BIGINT  NOT NULL,
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    volume    DOUBLE PRECISION,
    PRIMARY KEY (symbol, timeframe, ts_ms)
);

SELECT create_hypertable(
    'candles',
    'ts_ms',
    chunk_time_interval => 86400000,
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts
ON candles (symbol, timeframe, ts_ms DESC);
