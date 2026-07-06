"""Tests for `eesi.egnn`: radius graph and equivariance.

Runs as either pytest or a plain script:

    pytest tests/test_egnn.py
    python tests/test_egnn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from eesi.egnn import EGNN, radius_graph


# ---- helpers ---------------------------------------------------------------


def _make_model(
    d: int,
    r_cut: float,
    *,
    hidden: int = 16,
    n_layers: int = 3,
    k_attn: int = 1,
    n_global_tokens: int = 2,
    seed: int = 0,
) -> EGNN:
    torch.manual_seed(seed)
    net = EGNN(
        d=d, r_cut=r_cut,
        hidden=hidden, n_layers=n_layers, k_attn=k_attn,
        n_global_tokens=n_global_tokens,
    )
    return net.eval()


def _random_inputs(B: int, N: int, d: int, L: float, seed: int = 0):
    """Random (x, t) batch."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, N, d, generator=g) * L
    t = torch.rand(B, generator=g)
    return x, t


# ---- radius-graph correctness ----------------------------------------------


def test_radius_graph_matches_brute_force():
    """`radius_graph` agrees with a direct O(BN²) Python comparison."""
    torch.manual_seed(0)
    B, N, d, L, r_cut = 2, 8, 2, 10.0, 2.5
    x = torch.rand(B, N, d) * L

    edge_index, rel, d2 = radius_graph(x, r_cut)

    expected = set()
    expected_geometry = {}
    r2 = r_cut * r_cut
    for b in range(B):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff_ij = x[b, i] - x[b, j]
                d2_bij = float((diff_ij * diff_ij).sum())
                if d2_bij < r2:
                    src = b * N + j
                    dst = b * N + i
                    expected.add((src, dst))
                    expected_geometry[(src, dst)] = (d2_bij, diff_ij)

    got = set()
    got_geometry = {}
    E = edge_index.shape[1]
    for e in range(E):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        got.add((src, dst))
        got_geometry[(src, dst)] = (float(d2[e, 0]), rel[e])

    assert got == expected, f"edge set mismatch: missing {expected - got}, extra {got - expected}"
    for k in expected:
        d2_e, rel_e = expected_geometry[k]
        d2_g, rel_g = got_geometry[k]
        assert abs(d2_e - d2_g) < 1e-6
        assert torch.allclose(rel_e, rel_g, atol=1e-6)


def test_radius_graph_drops_self_loops_and_respects_cutoff():
    """No self-loops and every kept edge has `dist_sq < r_cut^2`."""
    torch.manual_seed(0)
    B, N, d, L, r_cut = 5, 15, 3, 19.0, 1.7
    x = torch.rand(B, N, d) * L
    edge_index, rel, d2 = radius_graph(x, r_cut)
    src, dst = edge_index[0], edge_index[1]
    assert (src != dst).all(), "self-loops present"
    assert (d2 < r_cut * r_cut + 1e-6).all(), "edge above cutoff slipped through"
    assert (src >= 0).all() and (src < B * N).all()
    assert (dst >= 0).all() and (dst < B * N).all()


def test_radius_graph_full_graph_at_large_cutoff():
    """When `r_cut` exceeds the box diameter the graph is fully connected."""
    torch.manual_seed(0)
    B, N, d, L = 4, 25, 4, 10.0
    # diameter of [0, L)^d is sqrt(d) * L
    diameter = (d ** 0.5) * L
    r_cut = diameter + 1.0
    x = torch.rand(B, N, d) * L
    edge_index, _, _ = radius_graph(x, r_cut)
    assert edge_index.shape[1] == B * N * (N - 1)


# ---- equivariance of EGNN -----------------------------


def _check_equiv(out_a: torch.Tensor, out_b: torch.Tensor, atol: float = 1e-4) -> None:
    assert torch.allclose(out_a, out_b, atol=atol), (
        f"mismatch: max |Δ| = {(out_a - out_b).abs().max().item():.3e}"
    )


def test_translation_invariance_of_v():
    """v is invariant under a global translation `x -> x + Δ`.

    Pairwise displacements `rel = coord[dst] - coord[src]` are translation-
    invariant, so the velocity `v = x_out - x_in` is also translation-invariant.
    """
    torch.manual_seed(0)
    B, N, d, L, r_cut = 2, 8, 3, 10.0, 3.0
    net = _make_model(d, r_cut)
    x, t = _random_inputs(B, N, d, L, seed=1)

    delta = torch.rand(1, 1, d)
    x_shift = x + delta

    out0 = net(t, x)
    out1 = net(t, x_shift)
    _check_equiv(out0, out1)


def test_permutation_equivariance_of_v():
    """v commutes with permutation of the particles."""
    torch.manual_seed(0)
    B, N, d, L, r_cut = 2, 8, 3, 10.0, 3.0
    net = _make_model(d, r_cut)
    x, t = _random_inputs(B, N, d, L, seed=2)

    g = torch.Generator().manual_seed(3)
    perm = torch.stack([torch.randperm(N, generator=g) for _ in range(B)])

    x_p = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, d))

    out0 = net(t, x)
    out1 = net(t, x_p)
    out0_p = torch.gather(out0, 1, perm.unsqueeze(-1).expand(-1, -1, d))
    _check_equiv(out0_p, out1)


def test_axis_swap_equivariance_of_v():
    """Swapping coordinate axes (x ↦ x[..., π]) swaps the same axes of v."""
    torch.manual_seed(0)
    B, N, d, L, r_cut = 2, 8, 4, 10.0, 3.0
    net = _make_model(d, r_cut)
    x, t = _random_inputs(B, N, d, L, seed=4)
    axis_perm = torch.tensor([3, 2, 0, 1])

    x_swap = x[..., axis_perm]
    out0 = net(t, x)
    out1 = net(t, x_swap)
    _check_equiv(out0[..., axis_perm], out1)


def test_sign_flip_equivariance_of_v():
    """Flipping coordinate signs flips the same signs of v."""
    torch.manual_seed(0)
    B, N, d, L, r_cut = 2, 8, 3, 10.0, 3.0
    net = _make_model(d, r_cut)
    x, t = _random_inputs(B, N, d, L, seed=5)

    sign = torch.tensor([1.0, -1.0, -1.0])
    x_flip = x * sign

    out0 = net(t, x)
    out1 = net(t, x_flip)
    _check_equiv(out0 * sign, out1)


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_radius_graph_matches_brute_force,
        test_radius_graph_drops_self_loops_and_respects_cutoff,
        test_radius_graph_full_graph_at_large_cutoff,
        test_translation_invariance_of_v,
        test_permutation_equivariance_of_v,
        test_axis_swap_equivariance_of_v,
        test_sign_flip_equivariance_of_v,
    ]
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
