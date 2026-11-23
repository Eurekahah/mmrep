"""P5 retrieval correctness against an independent full-sort reference."""
import numpy as np
import pytest

from eval.retrieval import recall_at_k


def reference_recall(images, texts, mapping, ks):
    # Deliberately simple full-matrix oracle, used only for tiny test inputs.
    images = images / np.linalg.norm(images, axis=1, keepdims=True)
    texts = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    scores = images @ texts.T
    i2t = np.argsort(-scores, axis=1, kind="stable")
    t2i = np.argsort(-scores.T, axis=1, kind="stable")
    result = {}
    for k in ks:
        result[f"i2t_R{k}"] = 100 * np.mean([any(mapping[j] == i for j in i2t[i, :k]) for i in range(len(images))])
        result[f"t2i_R{k}"] = 100 * np.mean([mapping[j] in t2i[j, :k] for j in range(len(texts))])
    return result


@pytest.mark.parametrize("block_size", [1, 7, 512])
def test_perfect_features_and_five_captions(block_size):
    images = np.eye(17, dtype=np.float32)
    mapping = np.repeat(np.arange(17), 5)
    texts = images[mapping]
    assert set(recall_at_k(images, texts, mapping, block_size=block_size).values()) == {100.0}


def test_shuffled_captions_and_nonfirst_positive():
    images = np.eye(5, dtype=np.float32)
    mapping = np.repeat(np.arange(5), 5)
    texts = -images[mapping]
    texts[4::5] = images
    order = np.random.default_rng(42).permutation(len(texts))
    result = recall_at_k(images, texts[order], mapping[order], block_size=2)
    assert result["i2t_R1"] == 100
    assert result == pytest.approx(reference_recall(images, texts[order], mapping[order], (1, 5, 10)))


@pytest.mark.parametrize("block_size", [1, 8, 512])
def test_blocked_scores_agree_with_independent_reference(block_size):
    rng = np.random.default_rng(25)
    images = rng.normal(size=(23, 19)).astype(np.float32)
    texts = rng.normal(size=(115, 19)).astype(np.float32)
    mapping = np.repeat(np.arange(23), 5)
    expected = reference_recall(images, texts, mapping, (1, 5, 10))
    actual = recall_at_k(images, texts, mapping, block_size=block_size)
    assert actual == pytest.approx(expected)
    for direction in ("i2t", "t2i"):
        assert actual[f"{direction}_R1"] <= actual[f"{direction}_R5"] <= actual[f"{direction}_R10"]


def test_scale_invariance():
    rng = np.random.default_rng(12)
    images, texts = rng.normal(size=(12, 7)), rng.normal(size=(60, 7))
    mapping = np.repeat(np.arange(12), 5)
    expected = recall_at_k(images, texts, mapping)
    actual = recall_at_k(images * rng.uniform(0.1, 20, (12, 1)),
                         texts * rng.uniform(0.1, 20, (60, 1)), mapping)
    assert actual == expected


def test_equal_scores_have_deterministic_index_tiebreak():
    images = np.ones((4, 3), dtype=np.float32)
    texts = np.ones((20, 3), dtype=np.float32)
    mapping = np.repeat(np.arange(4), 5)
    expected = reference_recall(images, texts, mapping, (1, 5, 10))
    assert recall_at_k(images, texts, mapping, block_size=1) == expected
    assert recall_at_k(images, texts, mapping, block_size=512) == expected


def test_small_candidate_sets_cap_k():
    assert set(recall_at_k(np.eye(2), np.repeat(np.eye(2), 5, axis=0),
                          np.repeat(np.arange(2), 5)).values()) == {100.0}


def test_independent_random_features_are_near_chance():
    rng = np.random.default_rng(42)
    n = 400
    result = recall_at_k(rng.normal(size=(n, 64)), rng.normal(size=(5 * n, 64)),
                         np.repeat(np.arange(n), 5))
    for direction in ("i2t", "t2i"):
        assert abs(result[f"{direction}_R1"] - 100 / n) < 1.5


@pytest.mark.parametrize("problem", ["zero", "nan", "dimension", "mapping_length", "mapping_range", "mapping_float", "missing_positive"])
def test_invalid_inputs_are_rejected(problem):
    images, texts = np.eye(3), np.repeat(np.eye(3), 5, axis=0)
    mapping = np.repeat(np.arange(3), 5)
    if problem == "zero":
        images[0] = 0
    elif problem == "nan":
        texts[0, 0] = np.nan
    elif problem == "dimension":
        texts = texts[:, :2]
    elif problem == "mapping_length":
        mapping = mapping[:-1]
    elif problem == "mapping_range":
        mapping[0] = 3
    elif problem == "mapping_float":
        mapping = mapping.astype(float)
    else:
        mapping[mapping == 2] = 1
    with pytest.raises(ValueError):
        recall_at_k(images, texts, mapping)


@pytest.mark.parametrize("kwargs", [{"ks": (0,)}, {"ks": ()}, {"ks": (1.5,)}, {"block_size": 0}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        recall_at_k(np.eye(2), np.eye(2), np.arange(2), **kwargs)
