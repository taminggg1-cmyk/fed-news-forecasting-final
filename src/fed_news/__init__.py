"""Fed News Forecasting, final study: does Federal Reserve tone help forecast FOMC decisions?

    config.py, http.py   paths, fixed design settings, cached HTTP
    download/            everything fetched from the internet -> dated tables in the data folder (ALFRED, FRED,
                         FOMC calendar, Fed documents, human labels, news indices)
    features/            Fed tone (ChronoBERT, Claude Haiku) and the training rows (per meeting, per month)
    models/              ordered logit, ridge, the linear walk-forward, LSTM / GRU / TFT building blocks
    evaluation/          metrics and tests (DM-HLN, Newey-West, McNemar, Holm)
    studies/             simple_model/ (Study 4) and networks/ (Study 4 version 2): the runs that produce results/

Start with docs/CODE_GUIDE.md: it shows where each piece of data comes from and which file reads it next.
"""
