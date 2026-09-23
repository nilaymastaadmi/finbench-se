"""Uncertainty scores and the metrics that judge them.

Discrete semantic entropy (Farquhar et al., Nature 2024): sample N answers, group them
into meaning clusters, and take the entropy of the cluster frequencies. High entropy means
the model gives different answers each time, which is the signal for a likely error.
"""
import math

from sklearn.metrics import roc_auc_score


def cluster_entropy(cluster_ids):
    """Entropy (nats) of the empirical distribution over cluster labels. Range [0, ln N]."""
    n = len(cluster_ids)
    assert n > 0
    counts = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    h = -sum((k / n) * math.log(k / n) for k in counts.values())
    assert -1e-12 <= h <= math.log(n) + 1e-12, h
    return max(h, 0.0)


def auroc_for_errors(scores, is_wrong):
    """AUROC of `scores` at ranking wrong answers above right ones. 0.5 = chance."""
    assert len(scores) == len(is_wrong)
    assert 0 < sum(is_wrong) < len(is_wrong), "AUROC needs both classes"
    return float(roc_auc_score([int(w) for w in is_wrong], scores))


def selective_accuracy(scores, is_correct, coverage):
    """Accuracy on the `coverage` fraction of questions with the LOWEST uncertainty score."""
    assert 0 < coverage <= 1
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    keep = order[:max(1, round(coverage * len(scores)))]
    return sum(is_correct[i] for i in keep) / len(keep)


def aurac(scores, is_correct, steps=100):
    """Area under the rejection-accuracy curve: mean selective accuracy over coverages 1/steps..1."""
    return sum(selective_accuracy(scores, is_correct, k / steps) for k in range(1, steps + 1)) / steps
