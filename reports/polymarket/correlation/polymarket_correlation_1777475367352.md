# Polymarket Correlation Research

- Rows: `5000`
- Horizons: `1h, 4h, 1d, 3d`

## Top Lead/Lag Correlations

- `tnx_ret_24h` -> `3d` corr=`-0.1831` samples=`4928` hit=`40.48%` top_bucket_avg=`-0.9104%`
- `tnx_trend_24h` -> `1d` corr=`-0.1668` samples=`4976` hit=`43.80%` top_bucket_avg=`-0.5419%`
- `tnx_trend_24h` -> `3d` corr=`-0.1589` samples=`4928` hit=`41.28%` top_bucket_avg=`-1.1015%`
- `gld_ret_1h` -> `3d` corr=`0.1274` samples=`4928` hit=`50.03%` top_bucket_avg=`-0.5347%`
- `gld_trend_24h` -> `3d` corr=`0.1193` samples=`4928` hit=`48.32%` top_bucket_avg=`-0.1125%`
- `uup_ret_1h` -> `3d` corr=`-0.0851` samples=`4928` hit=`48.75%` top_bucket_avg=`-0.3538%`
- `vix_ret_24h` -> `3d` corr=`-0.0801` samples=`4928` hit=`49.40%` top_bucket_avg=`-0.2831%`
- `tnx_ret_24h` -> `1d` corr=`-0.0695` samples=`4976` hit=`46.57%` top_bucket_avg=`-0.0327%`
- `gld_ret_24h` -> `3d` corr=`0.0645` samples=`4928` hit=`52.83%` top_bucket_avg=`-0.5106%`
- `qqq_trend_24h` -> `3d` corr=`0.0623` samples=`4928` hit=`47.97%` top_bucket_avg=`-0.2324%`
- `btc_ret_24h` -> `3d` corr=`-0.0622` samples=`4904` hit=`49.00%` top_bucket_avg=`-0.7248%`
- `spy_ret_24h` -> `3d` corr=`0.0590` samples=`4928` hit=`50.88%` top_bucket_avg=`0.5681%`
- `uup_trend_24h` -> `3d` corr=`-0.0565` samples=`4928` hit=`47.48%` top_bucket_avg=`-0.7287%`
- `gld_ret_1h` -> `1d` corr=`0.0547` samples=`4976` hit=`47.55%` top_bucket_avg=`-0.3438%`
- `qqq_ret_24h` -> `3d` corr=`0.0525` samples=`4928` hit=`51.83%` top_bucket_avg=`-0.2875%`
- `gld_trend_24h` -> `1d` corr=`0.0503` samples=`4976` hit=`47.99%` top_bucket_avg=`-0.0602%`
- `tnx_trend_24h` -> `4h` corr=`-0.0495` samples=`4996` hit=`48.71%` top_bucket_avg=`-0.0727%`
- `vix_trend_24h` -> `3d` corr=`-0.0444` samples=`4928` hit=`47.55%` top_bucket_avg=`-0.6756%`
- `btc_volatility_24h` -> `3d` corr=`0.0433` samples=`4904` hit=`46.57%` top_bucket_avg=`-0.6400%`
- `vix_ret_1h` -> `3d` corr=`-0.0407` samples=`4928` hit=`46.66%` top_bucket_avg=`-1.1971%`

## Strategy Hypotheses

No hypotheses met the sample threshold.

## Future Intelligence Hooks

- `macro_alignment_score`: use only after shadow/OOS validation.
- `social_confirmation_score`: use only after shadow/OOS validation.
- `news_risk_flag`: use only after shadow/OOS validation.

## Caveats

- Correlation is not causation.
- External feed delays and publication timing can degrade apparent signal.
- No social/news feature should create trades alone.
- Keep all outputs research/shadow-only until promotion evidence exists.