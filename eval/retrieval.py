"""Exact bidirectional Recall@K, in percent, with bounded query blocks."""
from __future__ import annotations

from numbers import Integral

import numpy as np

from .p5_common import as_numpy, unit_rows


def _topk(scores, k):
    """Partial selection; equal scores are consistently ordered by candidate index."""
    k = min(k, scores.shape[1])
    indices = np.argpartition(scores, scores.shape[1] - k, axis=1)[:, -k:]
    values = np.take_along_axis(scores, indices, axis=1)
    cutoff = values.min(axis=1)
    ambiguous = np.count_nonzero(scores == cutoff[:, None], axis=1) > np.count_nonzero(
        values == cutoff[:, None], axis=1)
    for row in np.flatnonzero(ambiguous):
        strict = indices[row][values[row] > cutoff[row]]
        tied = np.flatnonzero(scores[row] == cutoff[row])[:k - len(strict)]
        indices[row] = np.concatenate((strict, tied))
    values = np.take_along_axis(scores, indices, axis=1)
    order = np.lexsort((indices, -values), axis=1)
    return np.take_along_axis(indices, order, axis=1)


def recall_at_k(img, txt, txt2img, ks=(1, 5, 10), *, block_size=512):
    """Return six recalls in [0,100]; all captions are used in both directions.

    i2t succeeds if any retrieved caption maps to the query image. t2i succeeds
    if the mapped image is retrieved. K is capped at the available candidates.
    Both directions compute at most ``block_size`` query rows at once.
    """
    ks = tuple(ks)
    if not ks or any(not isinstance(k, Integral) or isinstance(k, bool) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    if not isinstance(block_size, Integral) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    images, texts = unit_rows(img, "img"), unit_rows(txt, "txt")
    if images.shape[1] != texts.shape[1]:
        raise ValueError("Image and text embedding dimensions differ")
    mapping = as_numpy(txt2img)
    if mapping.shape != (len(texts),) or mapping.dtype.kind not in "iu":
        raise ValueError("txt2img must be an integer array with one entry per text")
    if np.any(mapping < 0) or np.any(mapping >= len(images)):
        raise ValueError("txt2img contains an out-of-range image index")
    if len(np.unique(mapping)) != len(images):
        raise ValueError("Every image must have at least one positive caption")
    counts = {direction: {k: 0 for k in ks} for direction in ("i2t", "t2i")}
    for start in range(0, len(images), block_size):
        stop = min(start + block_size, len(images))
        top = _topk(images[start:stop] @ texts.T, max(ks))
        matches = mapping[top] == np.arange(start, stop)[:, None]
        for k in ks:
            counts["i2t"][k] += int(matches[:, :k].any(axis=1).sum())
    for start in range(0, len(texts), block_size):
        stop = min(start + block_size, len(texts))
        top = _topk(texts[start:stop] @ images.T, max(ks))
        matches = top == mapping[start:stop, None]
        for k in ks:
            counts["t2i"][k] += int(matches[:, :k].any(axis=1).sum())
    return {f"{direction}_R{k}": 100.0 * count / (len(images) if direction == "i2t" else len(texts))
            for direction, by_k in counts.items() for k, count in by_k.items()}
