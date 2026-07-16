"""Tests for `eesi.models.xygnn`: static chain graph, periodicity, EESI integration.

Runs as either pytest or a plain script:

    pytest tests/test_xygnn.py
    python tests/test_xygnn.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from eesi.interpolant import EESI
from eesi.models.xygnn import (
    XYChainGNN,
    angle_wrap,
    chain_edge_index,
    fourier_expand,
)


# ---- helpers ---------------------------------------------------------------


def _make_model(
    n_neighbors: int,
    *,
    hidden: int = 16,
    n_layers: int = 3,
    edge_order: int = 3,
    time_order: int = 3,
    seed: int = 0,
) -> XYChainGNN:
    torch.manual_seed(seed)
    net = XYChainGNN(
        n_neighbors=n_neighbors,
        hidden=hidden, n_layers=n_layers,
        edge_order=edge_order, time_order=time_order,
    )
    return net.eval()


def _random_angles(B: int, N: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, N, generator=g) * 2.0 * math.pi) - math.pi


# ---- helper correctness ----------------------------------------------------


def test_angle_wrap_range_and_identity():
    """`angle_wrap` maps into (-pi, pi] and is identity on that interval."""
    d = torch.linspace(-10.0, 10.0, 401)
    w = angle_wrap(d)
    assert (w <= math.pi + 1e-6).all() and (w > -math.pi - 1e-6).all()
    # wrapping is idempotent and differs from d by a multiple of 2*pi
    k = (d - w) / (2.0 * math.pi)
    assert torch.allclose(k, k.round(), atol=1e-5)


def test_fourier_expand_values():
    """`fourier_expand` returns [cos(kx)..., sin(kx)...] with the right shape."""
    x = torch.tensor([0.0, math.pi / 2])
    out = fourier_expand(x, order=2)
    assert out.shape == (2, 4)
    expected = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],                       # x = 0
        [0.0, -1.0, 1.0, 0.0],                      # x = pi/2: cos(pi/2),cos(pi),sin(pi/2),sin(pi)
    ])
    assert torch.allclose(out, expected, atol=1e-6)


# ---- static chain graph ----------------------------------------------------


def test_chain_edge_index_matches_brute_force():
    """Open-chain graph agrees with a direct enumeration; inv_dist = 1/k."""
    N, n_neighbors = 6, 2
    edge_index, inv_dist = chain_edge_index(N, n_neighbors)

    expected = {}
    for i in range(N):                              # i = dst
        for k in range(1, n_neighbors + 1):
            for j in (i - k, i + k):                # j = src
                if 0 <= j < N:
                    expected[(j, i)] = 1.0 / k

    got = {}
    for e in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        got[(src, dst)] = float(inv_dist[e, 0])

    assert got.keys() == expected.keys(), (
        f"edge mismatch: missing {set(expected) - set(got)}, extra {set(got) - set(expected)}"
    )
    for key in expected:
        assert abs(got[key] - expected[key]) < 1e-6


def test_chain_graph_no_self_loops_and_in_range():
    """No self-loops; indices in [0, N); interior nodes have 2*n_neighbors edges."""
    N, n_neighbors = 10, 3
    edge_index, _ = chain_edge_index(N, n_neighbors)
    src, dst = edge_index[0], edge_index[1]
    assert (src != dst).all()
    assert (src >= 0).all() and (src < N).all()
    assert (dst >= 0).all() and (dst < N).all()
    # an interior node (far from both ends) has n_neighbors edges on each side
    interior = N // 2
    assert int((dst == interior).sum()) == 2 * n_neighbors


# ---- backbone behaviour ----------------------------------------------------


def test_output_shape_and_finiteness():
    """Forward returns a field matching the input shape (both [B,N] and [B,N,1])."""
    N = 8
    net = _make_model(n_neighbors=2)
    x = _random_angles(3, N, seed=1)
    t = torch.rand(3)

    out = net(t, x)
    assert out.shape == (3, N)
    assert torch.isfinite(out).all()

    out3 = net(t, x.unsqueeze(-1))
    assert out3.shape == (3, N, 1)
    assert torch.allclose(out3.squeeze(-1), out, atol=1e-6)


def test_single_instance_handles_multiple_chain_lengths():
    """One model instance runs on inputs of different N without reconstruction."""
    net = _make_model(n_neighbors=2)
    t = torch.rand(3)
    for N in (5, 8, 13):
        out = net(t, _random_angles(3, N, seed=N))
        assert out.shape == (3, N) and torch.isfinite(out).all()


def test_hidden_and_time_order_are_decoupled():
    """`hidden` (MLP width) is independent of `time_order` (embedding size).

    The first-layer input width must be edge_attr_dim + 2*time_order, regardless
    of `hidden`, and forward must run for mismatched values of the two.
    """
    N, hidden, time_order, edge_order = 6, 5, 7, 2
    net = XYChainGNN(
        n_neighbors=2, hidden=hidden,
        n_layers=2, edge_order=edge_order, time_order=time_order,
    ).eval()

    first_linear = net.layers[0].net[0]
    expected_in = (1 + 2 * edge_order) + 2 * time_order
    assert first_linear.in_features == expected_in
    assert first_linear.out_features == hidden

    out = net(torch.rand(3), _random_angles(3, N, seed=20))
    assert out.shape == (3, N) and torch.isfinite(out).all()


def test_mlp_layers_controls_depth():
    """`mlp_layers` sets the number of hidden Linear(hidden, hidden) layers."""
    N = 6
    net1 = XYChainGNN(n_neighbors=2, hidden=8, n_layers=1, mlp_layers=1,
                      edge_order=2, time_order=2)
    net3 = XYChainGNN(n_neighbors=2, hidden=8, n_layers=1, mlp_layers=3,
                      edge_order=2, time_order=2)
    # count Linear layers inside a single conv MLP
    n_linear1 = sum(isinstance(m, torch.nn.Linear) for m in net1.layers[0].net)
    n_linear3 = sum(isinstance(m, torch.nn.Linear) for m in net3.layers[0].net)
    # mlp_layers hidden layers + 1 scalar readout
    assert n_linear1 == 1 + 1
    assert n_linear3 == 3 + 1

    out = net3.eval()(torch.rand(2), _random_angles(2, N, seed=21))
    assert out.shape == (2, N) and torch.isfinite(out).all()


def test_scalar_time_broadcasts():
    """A 0-dim time tensor is broadcast across the batch."""
    N = 7
    net = _make_model(n_neighbors=2)
    x = _random_angles(4, N, seed=2)
    out_scalar = net(torch.tensor(0.3), x)
    out_vector = net(torch.full((4,), 0.3), x)
    assert torch.allclose(out_scalar, out_vector, atol=1e-6)


def test_invariance_to_global_2pi_shift():
    """The velocity is unchanged when every angle is shifted by 2*pi.

    All edge features are wrapped differences and the output is theta - theta_in,
    so adding a multiple of 2*pi to the whole configuration cancels.
    """
    N = 8
    net = _make_model(n_neighbors=3)
    x = _random_angles(3, N, seed=3)
    t = torch.rand(3)
    out0 = net(t, x)
    out1 = net(t, x + 2.0 * math.pi)
    assert torch.allclose(out0, out1, atol=1e-4), (
        f"max |Δ| = {(out0 - out1).abs().max().item():.3e}"
    )


def test_invariance_to_per_node_2pi_shift():
    """Shifting individual angles by independent integer multiples of 2*pi is a no-op."""
    N = 8
    net = _make_model(n_neighbors=2)
    x = _random_angles(2, N, seed=4)
    t = torch.rand(2)
    g = torch.Generator().manual_seed(5)
    m = torch.randint(-2, 3, (2, N), generator=g).to(x.dtype)
    out0 = net(t, x)
    out1 = net(t, x + 2.0 * math.pi * m)
    assert torch.allclose(out0, out1, atol=1e-4), (
        f"max |Δ| = {(out0 - out1).abs().max().item():.3e}"
    )


def test_invariance_to_unit_time_shift():
    """The Fourier(2*pi*t) time embedding is periodic, so t and t+1 agree."""
    N = 6
    net = _make_model(n_neighbors=2)
    x = _random_angles(3, N, seed=6)
    t = torch.rand(3)
    out0 = net(t, x)
    out1 = net(t + 1.0, x)
    assert torch.allclose(out0, out1, atol=1e-4), (
        f"max |Δ| = {(out0 - out1).abs().max().item():.3e}"
    )


def test_backward_flows_to_all_parameters():
    """A scalar built from the output has a gradient for every parameter."""
    N = 6
    net = _make_model(n_neighbors=2)
    x = _random_angles(4, N, seed=7)
    t = torch.rand(4)
    net(t, x).square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


# ---- EESI integration ------------------------------------------------------


def test_eesi_loss_runs_and_backprops():
    """Two XYChainGNNs as (net_b, net_s) give a finite EESI loss that backprops."""
    torch.manual_seed(0)
    N, B = 8, 5
    net_b = XYChainGNN(n_neighbors=2, hidden=16, n_layers=2, edge_order=2, time_order=2)
    net_s = XYChainGNN(n_neighbors=2, hidden=16, n_layers=2, edge_order=2, time_order=2)
    model = EESI(net_b, net_s, d=N, path="linear", gamma="quad")

    x1 = _random_angles(B, N, seed=10)
    x0 = _random_angles(B, N, seed=11)
    losses = model.loss(x1, x0)
    total = losses["b"] + losses["s"]
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"


def test_eesi_ism_score_path_runs():
    """gamma='none' exercises the implicit-score-matching divergence path."""
    torch.manual_seed(0)
    N, B = 6, 4
    net_b = XYChainGNN(n_neighbors=2, hidden=16, n_layers=2, edge_order=2, time_order=2)
    net_s = XYChainGNN(n_neighbors=2, hidden=16, n_layers=2, edge_order=2, time_order=2)
    model = EESI(net_b, net_s, d=N, path="linear", gamma="none", n_hutchinson_probes=4)

    x1 = _random_angles(B, N, seed=12)
    x0 = _random_angles(B, N, seed=13)
    losses = model.loss(x1, x0)
    total = losses["b"] + losses["s"]
    assert torch.isfinite(total)
    total.backward()


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_angle_wrap_range_and_identity,
        test_fourier_expand_values,
        test_chain_edge_index_matches_brute_force,
        test_chain_graph_no_self_loops_and_in_range,
        test_output_shape_and_finiteness,
        test_single_instance_handles_multiple_chain_lengths,
        test_hidden_and_time_order_are_decoupled,
        test_mlp_layers_controls_depth,
        test_scalar_time_broadcasts,
        test_invariance_to_global_2pi_shift,
        test_invariance_to_per_node_2pi_shift,
        test_invariance_to_unit_time_shift,
        test_backward_flows_to_all_parameters,
        test_eesi_loss_runs_and_backprops,
        test_eesi_ism_score_path_runs,
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
