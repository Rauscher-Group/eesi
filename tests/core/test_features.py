"""Tests for `eesi.features`: the scalar-to-vector expansions network inputs use.

Runs as either pytest or a plain script:

    pytest tests/core/test_features.py
    python tests/core/test_features.py

These are three-line pure functions, so the bugs worth catching are not arithmetic
ones. They are:

  * a PERIOD error. `fourier_expand(2*pi*s, K)` has period exactly 1 and therefore
    maps s = 0 and s = 1 to the same vector -- the F1 bug from
    `plans/XY_MODEL_REVIEW.md`, which cost the XY net the ability to tell the two
    endpoints of the interpolant apart. `half_period_expand` exists so that the same
    mistake cannot be made again, for the interpolant time OR for TAP's chain index,
    where it would collapse the tail with the head of a directed polymer. The
    endpoint tests below are the whole reason this file exists.

  * a CHANNEL-LAYOUT error. Callers concatenate these blocks and then index them
    (`TAPDynamics._node_features` puts the raw scalar at channel 0 of each block so
    that order 0 reproduces its original features exactly), so the [cos..., sin...]
    ordering is load-bearing, not cosmetic.

  * a RE-EXPORT drift. `eesi.systems.xy.gnn` used to define these and now re-imports
    them; the last test pins that the two names are the same object, so a future
    edit cannot quietly reintroduce a second copy.

`order = 0` returning a width-0 tensor rather than raising is also pinned: it is what
lets `TAPDynamics` concatenate the expansion unconditionally, with no branch.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from eesi import features
from eesi.features import (
    cos_expand,
    fourier_expand,
    half_period_expand,
    time_features,
)
from eesi.systems.xy import gnn as xy_gnn


# ---- half_period_expand ----------------------------------------------------


def test_half_period_expand_values():
    """[cos(pi x), cos(2 pi x), sin(pi x), sin(2 pi x)] -- cosines first, then sines."""
    x = torch.tensor([0.0, 0.25, 0.6, 1.0], dtype=torch.float64)
    out = half_period_expand(x, order=2)
    assert out.shape == (4, 4)
    want = torch.stack([
        torch.cos(math.pi * x), torch.cos(2 * math.pi * x),
        torch.sin(math.pi * x), torch.sin(2 * math.pi * x),
    ], dim=-1)
    assert torch.allclose(out, want, atol=1e-15)


def test_half_period_expand_shapes_and_zero_order():
    """order = 0 gives a [..., 0] tensor that `cat`s cleanly, instead of raising.

    `TAPDynamics._node_features` relies on this: it concatenates the expansion with
    no `if`, so order 0 must be the identity on the feature vector rather than an
    error or a spurious column.
    """
    x = torch.rand(3, 5, dtype=torch.float64)
    for order in (0, 1, 4):
        assert half_period_expand(x, order).shape == (3, 5, 2 * order)
    empty = half_period_expand(x, 0)
    assert torch.cat([x.unsqueeze(-1), empty], dim=-1).shape == (3, 5, 1)


def test_half_period_expand_separates_the_endpoints():
    """s = 0 and s = 1 get DIFFERENT features -- the property full-period loses.

    A full-period expansion is 1-periodic, so cos(2 pi k * 0) = cos(2 pi k * 1) for
    every k and the two ends of [0, 1] are structurally indistinguishable. For the
    interpolant time that erases the endpoints where the drift and score differ most
    (plans/XY_MODEL_REVIEW.md, F1); for TAP's chain index it erases the difference
    between the tail and the head of a directed polymer. Half-period does not: the
    cosines alternate +-1 at the two ends. The second assert shows the contrast is
    real and not an artefact of the test's tolerance.
    """
    zero = torch.tensor(0.0, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    for order in (1, 2, 3, 4, 8):
        gap = (half_period_expand(zero, order) - half_period_expand(one, order)).abs().max()
        assert gap > 1.0, order
        # the mistake this function exists to prevent
        full = (fourier_expand(2 * math.pi * zero, order)
                - fourier_expand(2 * math.pi * one, order)).abs().max()
        assert full < 1e-12, order


def test_half_period_expand_sines_vanish_at_both_ends():
    """Documented behaviour: the cosines do all the endpoint discrimination."""
    for s in (0.0, 1.0):
        out = half_period_expand(torch.tensor(s, dtype=torch.float64), order=4)
        assert out[4:].abs().max() < 1e-12
        assert out[:4].abs().min() > 0.99


# ---- the moved functions ---------------------------------------------------


def test_moved_functions_still_behave():
    """`fourier_expand` / `cos_expand` / `time_features` survived the move intact.

    Their own behaviour is covered in depth by `tests/xy/test_gnn.py`, which imports
    them through the re-export; this is a shape-and-shape-only smoke check that they
    are importable from their new home.
    """
    x = torch.rand(6, dtype=torch.float64)
    assert fourier_expand(x, 3).shape == (6, 6)
    assert cos_expand(x, 3).shape == (6, 3)
    assert time_features(x, 0).shape == (6, 1)
    assert time_features(x, 2).shape == (6, 5)


def test_xy_reexports_are_the_same_objects():
    """`eesi.systems.xy.gnn` re-imports rather than redefines.

    `eesi.systems` forbids one system importing from another, which is why the
    expansions moved to the core in the first place; XY keeps the names bound so its
    own test module's imports still resolve. Identity, not equality: a second copy
    would pass any behavioural check on the day it was written and then drift.
    """
    assert xy_gnn.fourier_expand is features.fourier_expand
    assert xy_gnn.cos_expand is features.cos_expand
    assert xy_gnn.time_features is features.time_features


if __name__ == "__main__":
    tests = [
        test_half_period_expand_values,
        test_half_period_expand_shapes_and_zero_order,
        test_half_period_expand_separates_the_endpoints,
        test_half_period_expand_sines_vanish_at_both_ends,
        test_moved_functions_still_behave,
        test_xy_reexports_are_the_same_objects,
    ]
    failed = 0
    for t in tests:
        name = getattr(t, "__name__", "lambda")
        try:
            t()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
