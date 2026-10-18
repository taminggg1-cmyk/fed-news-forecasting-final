import pandas as pd
import pytest

from fed_news import config
from fed_news.collect import labels

S1 = "The committee decided to keep the target range unchanged this quarter."
S2 = "Inflation pressures warrant a further tightening of monetary policy soon."
S3 = "Participants noted that labor market conditions remained quite strong overall."


def _corpus(rows):
    df = pd.DataFrame(rows, columns=["doc_id", "type", "release", "text"])
    df["release"] = pd.to_datetime(df["release"])
    return df


def _labs(*sents, source="tdw"):
    return pd.DataFrame({"sentence": list(sents), "label": 0, "raw_year": 1999,
                         "source": source})


def test_date_labels_earliest_release_normalisation_and_drop():
    corpus = _corpus([
        ("late", "minutes", "2005-05-01", f"Intro text here. {S1} More."),
        ("early", "statement", "2001-02-01", f"{S1} Other text that is long enough here."),
        ("skip", "beige", "1990-01-01", S3),
    ])
    lab = _labs("  THE committee decided to keep   the target range unchanged this quarter. ",
                S2, S3)
    out = labels.date_labels(lab, corpus)
    assert list(out["sentence"]) == [labels._norm(lab["sentence"][0])]
    assert out["release"].iloc[0] == pd.Timestamp("2001-02-01")
    assert out["year"].iloc[0] == 2001
    assert list(out.columns) == ["sentence", "label", "release", "source", "year"]


def test_date_labels_substring_fallback():
    # label is a fragment that sentence splitting does not isolate
    frag = "keep the target range unchanged this quarter"
    corpus = _corpus([("a", "speech", "2003-01-01", S1), ("b", "speech", "2002-01-01", S1)])
    out = labels.date_labels(_labs(frag), corpus)
    assert out["release"].iloc[0] == pd.Timestamp("2002-01-01")


def test_load_labels_raises_below_match_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path)
    raw = _labs(S1, S2, S3, "unmatched one", "unmatched two")
    monkeypatch.setattr(labels, "_load_raw", lambda: raw)
    monkeypatch.setattr(labels, "build_corpus",
                        lambda: _corpus([("a", "statement", "2001-02-01", f"{S1} {S2} {S3}")]))
    with pytest.raises(ValueError, match="matched"):
        labels.load_labels()
    assert not (tmp_path / "labels.pkl").exists()


def test_load_labels_caches_when_rate_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path)
    raw = pd.concat([_labs(S1, S2), _labs(S3, source="wcb")], ignore_index=True)
    monkeypatch.setattr(labels, "_load_raw", lambda: raw)
    monkeypatch.setattr(labels, "build_corpus",
                        lambda: _corpus([("a", "statement", "2001-02-01", f"{S1} {S2} {S3}")]))
    out = labels.load_labels()
    assert len(out) == 3 and (tmp_path / "labels.pkl").exists()
    assert set(out["year"]) == {2001}


@pytest.mark.network
def test_real_tdw_match_rate_and_mapping():
    corpus_path = config.DATA / "corpus.pkl"
    if not corpus_path.exists():
        pytest.skip("corpus.pkl not built")
    from fed_news.collect.corpus import build_corpus
    raw = labels._load_raw()
    assert set(raw["label"]) == {0, 1, 2}
    assert (raw["sentence"] == raw["sentence"].str.lower()).all()
    dated = labels.date_labels(raw, build_corpus())
    for source in ("tdw", "wcb"):
        assert (dated["source"] == source).sum() / (raw["source"] == source).sum() >= labels.MIN_MATCH_RATE
    tdw = dated[dated["source"] == "tdw"]
    tight = tdw[tdw["sentence"].str.contains(r"\btighten", regex=True)]
    assert tight["label"].mode().iloc[0] == labels.LABELS["hawkish"]


def test_short_label_not_dated_by_fallback():
    corpus = _corpus([("a", "speech", "2001-01-01", f"{S1} The rate is up.")])
    out = labels.date_labels(_labs("the rate is up"), corpus)
    assert len(out) == 0


def test_stale_pickle_without_release_triggers_rebuild(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path)
    pd.DataFrame({"sentence": [S1], "label": [0], "year": [1990], "source": ["tdw"]}
                 ).to_pickle(tmp_path / "labels.pkl")
    raw = pd.concat([_labs(S1), _labs(S2, source="wcb")], ignore_index=True)
    monkeypatch.setattr(labels, "_load_raw", lambda: raw)
    monkeypatch.setattr(labels, "build_corpus",
                        lambda: _corpus([("a", "statement", "2001-02-01", f"{S1} {S2}")]))
    out = labels.load_labels()
    assert "release" in out.columns and set(out["year"]) == {2001}
    assert "release" in pd.read_pickle(tmp_path / "labels.pkl").columns


def test_norm_maps_entities_quotes_dashes_spaces():
    assert labels._norm("Fed&#8217;s  “goal”—price–stability now ok") \
        == "fed's \"goal\"-price-stability now ok"
    assert labels._norm("a &amp; b\n\tc") == "a & b c"
