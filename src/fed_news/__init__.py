"""Fed News Forecasting, final study: does Federal Reserve tone help forecast FOMC decisions?

    config.py, http.py   paths, fixed design settings, cached HTTP
    collect/             raw sources -> dated tables (ALFRED vintages, markets, FOMC calendar, Fed documents, labels)
    features/            stance scores (ChronoBERT, Claude Haiku) and the monthly / per-meeting feature panels
    models/              ordered logit, ridge, the linear walk-forward, LSTM / GRU / TFT building blocks
    evaluation/          metrics and tests (DM-HLN, Newey-West, McNemar, Holm)
    studies/             study4_haiku_logit (the simple model), study4_networks_v2 (the tuned networks)
"""
