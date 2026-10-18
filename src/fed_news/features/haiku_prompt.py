"""BUILD helper: what is sent to Claude Haiku, and how its reply is read.

The system prompt, the numbered sentence list, the reply parser, and the masking that hides years, months, numbers,
names and events. Used by features/haiku_tone.py.

Details: The Claude Haiku 5.5 labelling prompt, reply parser and sentence masking used to build the Haiku index
(features/haiku_tone.py). Copied unchanged from the earlier labelling pilot (fed_news/studies/poc/poc_llm_labels.py)
and leak test (fed_news/studies/poc/poc_haiku_leak.py) in fed-news-forecasting ad2183c.
"""
import json
import os
import re

import pandas as pd

from fed_news import config

MODEL = "claude-haiku-5-5"
CODES = {"D": 0, "H": 1, "N": 2}  # labels.LABELS: dovish 0, hawkish 1, neutral 2

SYSTEM = (
    "You classify sentences from US Federal Reserve communications by their monetary policy stance.\n"
    "H (hawkish): the sentence points toward tighter monetary policy - higher or rising interest rates, less "
    "accommodation, concern that inflation is too high or rising, or an economy or labour market that is strong "
    "or overheating.\n"
    "D (dovish): the sentence points toward looser monetary policy - lower interest rates or more accommodation, "
    "inflation that is low or falling, or an economy or labour market that is weak, slowing or under stress.\n"
    "N (neutral): neither - descriptive or procedural, balanced, or not about the policy outlook.\n"
    "Judge only the text of each sentence. Reply with one JSON object mapping each sentence number to H, D or N, "
    "and nothing else."
)


def batch_prompt(sentences: list[str]) -> str:
    """Number the sentences, one per line, for one request."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))


def parse_labels(text: str, n: int) -> list[int]:
    """Read Haiku's JSON reply into label codes (0 dovish, 1 hawkish, 2 neutral); raise if any is missing."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    obj = {str(k).strip(): str(v).strip().upper() for k, v in json.loads(m.group(0)).items()}
    out = []
    for i in range(1, n + 1):
        v = obj.get(str(i))
        if v not in CODES:
            raise ValueError(f"sentence {i}: label {v!r}")
        out.append(CODES[v])
    return out


def _client():
    """Anthropic API client, using ANTHROPIC_API_KEY or <data>/secrets/anthropic_api_key.txt."""
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    path = config.SECRETS / "anthropic_api_key.txt"
    if not key and path.exists():
        key = path.read_text().strip()
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


ATTEMPTS = {  # design doc: attempt 1 (no thinking, effort low); attempt 2 (adaptive thinking, effort medium)
    1: {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}, "max_tokens": 1024},
    2: {"output_config": {"effort": "medium"}, "max_tokens": 8000},
}

KNOWN_NAMES = {"greenspan", "bernanke", "yellen", "powell"}
NOT_NAMES = {"affairs"}  # last word of a job-title speaker field, not a surname

_EVENTS = re.compile(r"\b(lehman(?: brothers)?|bear stearns|aig|fannie mae|freddie mac|katrina|rita|september 11|9/11"
                     r"|enron|worldcom|y2k|covid(?:-19)?|coronavirus|pandemic|omicron|brexit|ukraine"
                     r"|silicon valley bank|svb)\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH = re.compile(r"\b(?:january|february|april|june|july|august|september|october|november|december)\b")
_AMBIG_BEFORE = re.compile(r"\b(in|since|during|through|until|by|of|early|late|mid|from|last|next|between)\s+"
                           r"(?:may|march)\b")
_AMBIG_AFTER = re.compile(r"\b(?:may|march)(?=\s+(?:\[|\d))")
_NUM = re.compile(r"\d[\d\-/.,]*\d|\d")


def official_names(corpus: pd.DataFrame) -> set[str]:
    """Surnames to mask: known Fed chairs plus every speaker in the corpus."""
    names = set(KNOWN_NAMES)
    for s in corpus["speaker"].dropna().unique():
        parts = [p for p in re.split(r"[\s,]+", s.strip()) if p and p.rstrip(".") not in ("Jr", "Sr")]
        if parts:
            names.add(parts[-1].lower().rstrip("."))
    return names - NOT_NAMES


def mask(sentence: str, names: set[str]) -> str:
    """Hide events, names, years, months and numbers in a sentence."""
    s = _EVENTS.sub("[EVENT]", sentence.lower())
    if names:
        s = re.sub(r"\b(?:" + "|".join(sorted(map(re.escape, names))) + r")\b", "[NAME]", s)
    s = _YEAR.sub("[YEAR]", s)
    s = _MONTH.sub("[MONTH]", s)
    s = _AMBIG_BEFORE.sub(lambda m: f"{m.group(1)} [MONTH]", s)
    s = _AMBIG_AFTER.sub("[MONTH]", s)
    return _NUM.sub("[NUM]", s)
