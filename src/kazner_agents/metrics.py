"""Scoring, the same way as `kazner.evaluate.compute_metrics` (seqeval, default mode).

Entity level: an entity counts as correct only if its type AND both boundaries match.
Overall scores are micro-averaged over all entities of all sentences.
"""

from __future__ import annotations

from collections.abc import Sequence

from seqeval.metrics import accuracy_score, classification_report
from seqeval.metrics.sequence_labeling import get_entities


def compute_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> dict:
    """Entity-level precision / recall / F1, token accuracy and per-type scores."""
    if len(gold) != len(pred) or any(len(g) != len(p) for g, p in zip(gold, pred)):
        raise ValueError("gold and pred must have the same sentences and lengths")
    gold, pred = [list(s) for s in gold], [list(s) for s in pred]
    n_gold, n_pred = len(get_entities(gold)), len(get_entities(pred))
    if n_gold == 0 and n_pred == 0:  # nothing to find and nothing found: seqeval has no rows
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
                "n_gold_entities": 0, "n_pred_entities": 0, "per_type": {}}
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    overall = report.pop("micro avg")
    report.pop("macro avg")
    report.pop("weighted avg")
    return {
        "precision": float(overall["precision"]),
        "recall": float(overall["recall"]),
        "f1": float(overall["f1-score"]),
        "accuracy": float(accuracy_score(gold, pred)),
        "n_gold_entities": n_gold,
        "n_pred_entities": n_pred,
        "per_type": {
            name: {
                "precision": float(s["precision"]),
                "recall": float(s["recall"]),
                "f1": float(s["f1-score"]),
                "support": int(s["support"]),
            }
            for name, s in sorted(report.items())
        },
    }


def agreement(first: Sequence[Sequence[str]], second: Sequence[Sequence[str]]) -> dict:
    """How much two annotations agree when there is no gold.

    entity_f1: F1 of one against the other (the same whichever is taken as reference);
    token_agreement: share of words with the same label.
    """
    scores = compute_metrics(first, second)
    words = sum(len(s) for s in first)
    same = sum(a == b for s1, s2 in zip(first, second) for a, b in zip(s1, s2))
    return {"entity_f1": scores["f1"], "token_agreement": same / words if words else 1.0}
