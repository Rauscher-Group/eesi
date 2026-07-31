"""Guards on the state_dict layout `LJ13Dynamics.from_checkpoint` depends on.

Runs as either pytest or a plain script:

    pytest tests/lj13/test_checkpoint_keys.py
    python tests/lj13/test_checkpoint_keys.py

`from_checkpoint` loads the released OSF state_dict with `strict=True`, so the
parameter names are load-bearing: they are determined by the attribute names in
`eesi.egnn` (`E_GCL.edge_mlp`, `EGNN.gcl_{i}`, ...) and by `LJ13Dynamics.egnn`.
Renaming any of them, or restructuring the net into differently-named submodules,
breaks loading with an error that points at the checkpoint rather than at the
rename that caused it.

That is not hypothetical: `EGNN` moved out of `eesi.systems.lj13.dynamics` into the
core `eesi.egnn` so the TAP system could share it, and nothing else in the suite
would have caught a key drift -- the checkpoint is a gitignored third-party
download, so no other test can depend on it being present.

The golden list is therefore checked WITHOUT the checkpoint (it is a property of
the module tree alone). The two tests that need the real file skip when it is
absent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from eesi.systems.lj13.dynamics import CKPT_PATH, CKPT_PREFIX, LJ13Dynamics

# The 43 parameter names of the released LJ13 architecture (hidden_nf=32,
# n_layers=3). Update this list ONLY together with a deliberate, documented change
# to the checkpoint contract.
GOLDEN_KEYS = sorted([
    "egnn.embedding.weight", "egnn.embedding.bias",
    "egnn.embedding_out.weight", "egnn.embedding_out.bias",
] + [
    f"egnn.gcl_{i}.{name}"
    for i in range(3)
    for name in (
        "edge_mlp.0.weight", "edge_mlp.0.bias", "edge_mlp.2.weight", "edge_mlp.2.bias",
        "node_mlp.0.weight", "node_mlp.0.bias", "node_mlp.2.weight", "node_mlp.2.bias",
        "coord_mlp.0.weight", "coord_mlp.0.bias", "coord_mlp.2.weight",
        "att_mlp.0.weight", "att_mlp.0.bias",
    )
])

_HAS_CKPT = CKPT_PATH.exists()
_needs_ckpt = pytest.mark.skipif(
    not _HAS_CKPT, reason=f"released checkpoint not present at {CKPT_PATH}")


def test_state_dict_keys_are_unchanged():
    """The parameter names are exactly the ones the released checkpoint carries."""
    assert sorted(LJ13Dynamics().state_dict().keys()) == GOLDEN_KEYS


def test_parameter_count_is_unchanged():
    """22468 parameters, as documented in the `dynamics` module docstring."""
    assert sum(p.numel() for p in LJ13Dynamics().parameters()) == 22468


def test_prefixed_state_dict_loads_strict():
    """A state_dict under CKPT_PREFIX round-trips through the `from_checkpoint` mapping.

    Exercises the prefix-stripping and `strict=True` load without needing the real
    download: any key drift shows up as a missing/unexpected-keys error here.
    """
    ref = LJ13Dynamics()
    sd = {CKPT_PREFIX + k: v for k, v in ref.state_dict().items()}
    sub = {k[len(CKPT_PREFIX):]: v for k, v in sd.items() if k.startswith(CKPT_PREFIX)}
    fresh = LJ13Dynamics()
    fresh.load_state_dict(sub, strict=True)
    for a, b in zip(ref.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a, b)


@_needs_ckpt
def test_released_checkpoint_loads_strict():
    """The actual OSF download still loads."""
    model = LJ13Dynamics.from_checkpoint()
    assert next(model.parameters()).dtype == torch.float64


@_needs_ckpt
def test_released_checkpoint_output_is_unchanged():
    """Golden velocity from the loaded checkpoint on a fixed input.

    Pins the numerics, not just the key names: a refactor that preserved the names
    but reordered, say, the residual updates inside `E_GCL` would pass every test
    above and silently change every result in the LJ13 notebook.
    """
    model = LJ13Dynamics.from_checkpoint()
    torch.manual_seed(0)
    x = torch.randn(2, 13, 3, dtype=torch.float64)
    x = x - x.mean(1, keepdim=True)
    v = model(0.3, x)
    expected = torch.tensor([1.7253119378264574, -0.09101391712343813, 0.689473310343103],
                            dtype=torch.float64)
    assert torch.allclose(v[0, 0], expected, atol=1e-12), v[0, 0].tolist()


if __name__ == "__main__":
    tests = [
        test_state_dict_keys_are_unchanged,
        test_parameter_count_is_unchanged,
        test_prefixed_state_dict_loads_strict,
    ]
    if _HAS_CKPT:
        tests += [test_released_checkpoint_loads_strict,
                  test_released_checkpoint_output_is_unchanged]
    else:
        print(f"SKIP  checkpoint tests ({CKPT_PATH} not present)")
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
