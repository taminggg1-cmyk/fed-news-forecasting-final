"""Metrics, the pre-registered H1 decision, and RESULTS.md (spec section 8)."""
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from fed_news import config
from fed_news.evaluation.stats_tests import dm_hln, holm, mcnemar_greater

PAPER_ROWS = [  # spec ladder row 1: the paper's own reported numbers
    ("paper LSTM (Table 2)", 0.2376 ** 2, 0.2376, 0.1495, -0.0006),
    ("paper GRU (Table 2)", 0.2705 ** 2, 0.2705, 0.1327, 0.0091),
    ("paper LLM, lookahead (Table 3)", 0.0311, 0.1762, 0.1483, np.nan),
]
NAMES = ["rw", "taylor", "S", "S+N", "placebo", "shuffled", "S+M", "S+M+N"]


def metrics(preds: pd.DataFrame) -> dict:
    err = preds["y_pred"] - preds["y_true"]
    return {"n": len(preds), "mse": float(np.mean(err ** 2)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "mbe": float(np.mean(err)),
            "accuracy": float((preds["class_pred"] == preds["class_true"]).mean()),
            "macro_f1": float(f1_score(preds["class_true"], preds["class_pred"], average="macro"))}


def decide(main_p, placebo_p, shuffled_p, alpha: float = config.ALPHA, shuffled_alpha: float = 0.01) -> str:
    for p in list(main_p) + list(placebo_p) + list(shuffled_p):
        if np.isnan(p):
            raise ValueError("NaN p-value in decision inputs")
    if any(p < alpha for p in placebo_p) or any(p < shuffled_alpha for p in shuffled_p):
        return "void"
    return "supported" if all(p < alpha for p in main_p) else "not supported"


def deviations_section(repo=config.REPO) -> str:
    path = repo / "DEVIATIONS.md"
    return path.read_text(encoding="utf-8").strip() if path.exists() else "None recorded."


def _git(repo, *args) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def provenance_section(repo=config.REPO) -> str:
    prereg = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{config.PREREG_TAG}^{{commit}}") or "no tag"
    diff = _git(repo, "diff", "--stat", config.PREREG_TAG, "--", "src") if prereg != "no tag" else ""
    return "\n".join([f"- prereg tag commit: {prereg}", f"- HEAD: {_git(repo, 'rev-parse', 'HEAD')}",
                      "- `git diff --stat prereg -- src`:", "", "```", diff, "```"])


def _load(task):
    return {n: pd.read_csv(config.RESULTS / f"preds_{task}_{n}.csv", parse_dates=["date"]) for n in NAMES}


def _window(p, window):
    return p[(p["date"] >= window[0]) & (p["date"] <= window[1])] if window else p


def _sq(p):
    return ((p["y_pred"] - p["y_true"]) ** 2).to_numpy()


def _correct(p):
    return (p["class_pred"] == p["class_true"]).to_numpy()


def _three_tests(t1, t2, new, base="S"):
    # Check T1 dates match
    if not t1[base]["date"].equals(t1[new]["date"]):
        raise ValueError(f"prediction dates differ: {new} vs {base}")
    # Check T2 dates match
    if not t2[base]["date"].equals(t2[new]["date"]):
        raise ValueError(f"prediction dates differ: {new} vs {base}")

    p1 = dm_hln(_sq(t1[base]), _sq(t1[new]))[1]
    p2 = dm_hln(_sq(t2[base]), _sq(t2[new]))[1]
    p3 = mcnemar_greater(_correct(t2[base]), _correct(t2[new]))[2]
    return [p1, p2, p3]


def main() -> None:
    t1, t2 = _load("T1"), _load("T2")
    rows = []
    for task, preds, paper in (("T1", t1, config.PAPER_T1), ("T2", t2, config.PAPER_T2)):
        for window_name, window in (("primary", None), ("paper window", paper)):
            for name in NAMES:
                rows.append({"task": task, "window": window_name, "model": name,
                             **metrics(_window(preds[name], window))})
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS / "metrics.csv", index=False)

    main_p, placebo_p = _three_tests(t1, t2, "S+N"), _three_tests(t1, t2, "placebo")
    shuffled_p = _three_tests(t1, t2, "shuffled", base="rw")
    verdict = decide(main_p, placebo_p, shuffled_p)
    tests = pd.DataFrame({"test": ["T1 DM-HLN", "T2 DM-HLN", "T2 McNemar"],
                          "S+N vs S": main_p, "placebo vs S": placebo_p,
                          "shuffled vs rw": shuffled_p})
    tests.to_csv(config.RESULTS / "h1_tests.csv", index=False)

    h2 = {"H2c T1 S+N vs rw": dm_hln(_sq(t1["rw"]), _sq(t1["S+N"]))[1],
          "H2c T2 S+N vs rw": dm_hln(_sq(t2["rw"]), _sq(t2["S+N"]))[1],
          "H2d T1 S+M+N vs S+M": dm_hln(_sq(t1["S+M"]), _sq(t1["S+M+N"]))[1],
          "H2d T2 S+M+N vs S+M": dm_hln(_sq(t2["S+M"]), _sq(t2["S+M+N"]))[1],
          "H2d T2 McNemar S+M+N vs S+M": mcnemar_greater(_correct(t2["S+M"]), _correct(t2["S+M+N"]))[2]}
    h2_frame = pd.DataFrame({"hypothesis": list(h2), "p": list(h2.values()),
                             "holm_within_plan1": [holm(h2)[k] for k in h2]})
    h2_frame.to_csv(config.RESULTS / "h2_pvalues.csv", index=False)

    paper = pd.DataFrame(PAPER_ROWS, columns=["model", "mse", "rmse", "mae", "mbe"])
    stance_acc = pd.read_csv(config.RESULTS / "stance_accuracy.csv")
    lines = [
        "# Results: Plan 1 (H1)", "",
        f"**H1 verdict: {verdict}.**", "",
        "## Pre-registered tests (one-sided p)", "", tests.to_markdown(index=False, floatfmt=".4f"), "",
        "## Metrics", "", table.to_markdown(index=False, floatfmt=".4f"), "",
        "## The paper's reported numbers (T1 window Jan 2016 - Apr 2025; LLM has lookahead)", "",
        paper.to_markdown(index=False, floatfmt=".4f"), "",
        "## Secondary (Holm within Plan 1 only; final Holm after Plans 2-3)", "",
        h2_frame.to_markdown(index=False, floatfmt=".4f"), "",
        "## Stance classifier held-out accuracy", "", stance_acc.to_markdown(index=False, floatfmt=".3f"), "",
        "## Deviations", "", deviations_section(), "",
        "## Provenance", "", provenance_section(), "",
    ]
    config.report_path("RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
