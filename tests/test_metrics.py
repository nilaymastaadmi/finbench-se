import math

import pytest

from fse.metrics import aurac, auroc_for_errors, cluster_entropy, selective_accuracy


def test_entropy_zero_when_all_samples_agree():
    assert cluster_entropy([0, 0, 0, 0, 0]) == 0.0


def test_entropy_max_when_every_sample_differs():
    assert cluster_entropy([0, 1, 2, 3, 4]) == pytest.approx(math.log(5))


def test_entropy_two_clusters():
    # 3 vs 2 split: -(0.6 ln 0.6 + 0.4 ln 0.4)
    assert cluster_entropy(["a", "a", "a", "b", "b"]) == pytest.approx(0.6730116670092565)


def test_auroc_perfect_and_inverted_and_chance():
    wrong = [0, 0, 1, 1]
    assert auroc_for_errors([0.1, 0.2, 0.8, 0.9], wrong) == 1.0
    assert auroc_for_errors([0.9, 0.8, 0.2, 0.1], wrong) == 0.0
    assert auroc_for_errors([0.5, 0.5, 0.5, 0.5], wrong) == 0.5


def test_auroc_refuses_single_class():
    with pytest.raises(AssertionError):
        auroc_for_errors([0.1, 0.2], [0, 0])


def test_selective_accuracy_keeps_lowest_uncertainty():
    scores = [0.0, 0.1, 0.9, 1.0]
    correct = [1, 1, 0, 0]
    assert selective_accuracy(scores, correct, 0.5) == 1.0
    assert selective_accuracy(scores, correct, 1.0) == 0.5


def test_aurac_beats_full_coverage_only_when_score_is_informative():
    correct = [1, 1, 0, 0]
    good = aurac([0.0, 0.1, 0.9, 1.0], correct)
    bad = aurac([1.0, 0.9, 0.1, 0.0], correct)
    assert good > 0.5 > bad


def test_rrf_rewards_agreement_between_rankings():
    from fse.hybrid import rrf
    # id 7 is 2nd in both lists and must beat ids that top only one list
    assert rrf([[1, 7, 2], [3, 7, 4]])[0] == 7
    assert set(rrf([[1], [2]])) == {1, 2}
