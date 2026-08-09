"""Tests for `eesi.systems.lj13.data.load_ref_data`'s random subsetting.

Runs as either pytest or a plain script:

    pytest tests/lj13/test_lj13_data.py
    python tests/lj13/test_lj13_data.py

Everything here runs on a small synthetic `.npy` in tmp_path -- the real OSF file is
a 1.5 GB third-party download, so no test may depend on it. What is being pinned is
that `n` draws a random subset (not the leading block, which on MCMC data is a short
correlated window), that the draw is reproducible from `seed`, and that the rows
returned are genuinely rows of the file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch

from eesi.systems.lj13.data import load_ref_data


M = 200  # rows in the synthetic file


def _write(tmp_path: Path, m: int = M) -> Path:
    """A synthetic reference file, (m, 39). Gaussian rows are distinct with
    probability 1, so a returned configuration identifies its source row."""
    p = tmp_path / "ref.npy"
    np.save(p, np.random.default_rng(0).normal(size=(m, 39)))
    return p


def _source_rows(x: torch.Tensor, full: torch.Tensor) -> list[int]:
    """Indices in `full` of the configurations in `x`. Exact equality is the right
    test: both went through the same centering arithmetic on the same bytes."""
    return [int((full == c).all(-1).all(-1).nonzero()[0]) for c in x]


def test_subset_is_random_and_not_the_leading_block(tmp_path):
    p = _write(tmp_path)
    full = load_ref_data(p)
    idx = _source_rows(load_ref_data(p, n=20), full)

    assert len(set(idx)) == 20, "drawn with replacement -- rows repeat"
    assert idx != list(range(20)), "still returning the leading block"
    assert max(idx) > 20, "the draw never reaches past the head of the file"
    assert idx == sorted(idx), "documented contract: rows come back in file order"


def test_seed_makes_the_draw_reproducible(tmp_path):
    p = _write(tmp_path)
    assert torch.equal(load_ref_data(p, n=20), load_ref_data(p, n=20)), \
        "the default seed must be deterministic"
    assert torch.equal(load_ref_data(p, n=20, seed=3), load_ref_data(p, n=20, seed=3))
    assert not torch.equal(load_ref_data(p, n=20, seed=3), load_ref_data(p, n=20, seed=4))
    # An explicit Generator is accepted too, and advances between calls.
    rng = np.random.default_rng(0)
    assert not torch.equal(load_ref_data(p, n=20, seed=rng), load_ref_data(p, n=20, seed=rng))


def test_seed_none_is_a_fresh_draw(tmp_path):
    p = _write(tmp_path)
    draws = {tuple(_source_rows(load_ref_data(p, n=20, seed=None), load_ref_data(p)))
             for _ in range(5)}
    assert len(draws) > 1, "seed=None must not be pinned to one subset"


def test_shape_dtype_and_com_free(tmp_path):
    p = _write(tmp_path)
    x = load_ref_data(p, n=20)
    assert x.shape == (20, 13, 3) and x.dtype == torch.float64
    assert x.mean(1).abs().max() < 1e-12
    assert load_ref_data(p, n=20, dtype=torch.float32).dtype == torch.float32


def test_whole_file_needs_no_subsetting(tmp_path):
    """`n=None` and `n >= len(file)` both return everything, in file order."""
    p = _write(tmp_path)
    full = load_ref_data(p)
    assert full.shape == (M, 13, 3)
    assert torch.equal(load_ref_data(p, n=M), full)
    assert torch.equal(load_ref_data(p, n=M + 50), full), "asking too much must not raise"
    assert _source_rows(full, full) == list(range(M))


def test_missing_file_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="LJ13 reference data not found"):
        load_ref_data(tmp_path / "nope.npy")


if __name__ == "__main__":
    import tempfile

    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn(Path(tempfile.mkdtemp()))
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print("all good" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
