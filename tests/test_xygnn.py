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
    cos_expand,
    fourier_expand,
    time_features,
)


# ---- helpers ---------------------------------------------------------------


def _make_model(
    n_neighbors: int,
    *,
    hidden: int = 16,
    n_layers: int = 3,
    edge_order: int = 3,
    time_order: int = 0,
    time_dim: int = 8,
    seed: int = 0,
) -> XYChainGNN:
    torch.manual_seed(seed)
    net = XYChainGNN(
        n_neighbors=n_neighbors,
        hidden=hidden, n_layers=n_layers,
        edge_order=edge_order, time_order=time_order, time_dim=time_dim,
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


def test_cos_expand_values_and_parity():
    """`cos_expand` returns [cos(kx)] and is exactly even under x -> -x."""
    x = torch.tensor([0.0, math.pi / 2])
    out = cos_expand(x, order=2)
    assert out.shape == (2, 2)
    expected = torch.tensor([[1.0, 1.0], [0.0, -1.0]])   # x=0; x=pi/2: cos(pi/2),cos(pi)
    assert torch.allclose(out, expected, atol=1e-6)

    y = torch.linspace(-4.0, 4.0, 33).double()
    assert torch.equal(cos_expand(y, 4), cos_expand(-y, 4))


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


def test_chain_edge_index_is_cached():
    """Repeated calls return the SAME tensors, not a rebuilt copy.

    The graph is static, so the Python build loop must not run once per forward.
    Identity (not equality) is the assertion that catches a regression here.
    """
    a_ei, a_inv = chain_edge_index(6, 2)
    b_ei, b_inv = chain_edge_index(6, 2)
    assert a_ei is b_ei and a_inv is b_inv
    # a different key still builds its own graph
    c_ei, _ = chain_edge_index(6, 3)
    assert c_ei is not a_ei and c_ei.shape[1] > a_ei.shape[1]


def test_batch_graph_is_cached_and_matches_uncached():
    """`_batch_graph` is cached too, and still tiles the single-chain graph."""
    net = _make_model(n_neighbors=2)
    B, N = 3, 6
    ei, inv, batch = net._batch_graph(B, N)
    ei2, inv2, batch2 = net._batch_graph(B, N)
    assert ei is ei2 and inv is inv2 and batch is batch2

    ei1, inv1 = chain_edge_index(N, 2)
    E1 = ei1.shape[1]
    assert ei.shape == (2, B * E1) and batch.shape == (B * E1,)
    for b in range(B):
        blk = ei[:, b * E1:(b + 1) * E1]
        assert torch.equal(blk, ei1 + b * N)
        assert torch.equal(inv[b * E1:(b + 1) * E1], inv1)
        assert (batch[b * E1:(b + 1) * E1] == b).all()


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


def test_hidden_and_time_dim_are_decoupled():
    """`hidden` (MLP width) is independent of `time_dim` (embedding size).

    The first-layer input width must be edge_attr_dim + time_dim, regardless of
    `hidden`, and forward must run for mismatched values of the two. `time_order`
    sets only the width of the time MLP's *input*, not of its output.
    """
    N, hidden, time_order, time_dim, edge_order = 6, 5, 7, 9, 2
    net = XYChainGNN(
        n_neighbors=2, hidden=hidden, n_layers=2,
        edge_order=edge_order, time_order=time_order, time_dim=time_dim,
    ).eval()

    first_linear = net.layers[0].net[0]
    expected_in = (1 + edge_order) + time_dim       # inv_dist + cos harmonics
    assert first_linear.in_features == expected_in
    assert first_linear.out_features == hidden
    assert net.time_mlp[0].in_features == 1 + 2 * time_order
    assert net.time_mlp[-1].out_features == time_dim

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


def test_time_features_are_injective_at_the_endpoints():
    """t=0 and t=1 must not share an embedding, at any `time_order`.

    The old embedding was `fourier_expand(2*pi*t, K)`, exactly 1-periodic, so the
    two endpoints mapped to the identical vector -- see plans/XY_MODEL_REVIEW.md
    F1, where the trained nets were measured to be bitwise identical there.
    """
    t = torch.tensor([0.0, 1.0])
    for order in (0, 8):
        f = time_features(t, order)
        assert f.shape == (2, 1 + 2 * order)
        assert (f[0] - f[1]).abs().max() > 0.5, f"collapsed at time_order={order}"


def test_network_separates_the_time_endpoints():
    """End to end: net(0, x) and net(1, x) differ substantially on identical x."""
    N = 6
    x = _random_angles(3, N, seed=6)
    for order in (0, 8):
        net = _make_model(n_neighbors=2, time_order=order)
        v0, v1 = net(torch.zeros(3), x), net(torch.ones(3), x)
        rel = ((v0 - v1).norm() / v1.norm()).item()
        assert rel > 1e-2, f"endpoints collapsed at time_order={order}: rel = {rel:.3e}"


# ---- symmetry: the velocity field must transform under O(2) x Z2^site --------
#
# Both marginals are invariant under G = {theta -> +-theta + phi} x {site reversal}
# (review Sec. 6.1), and the regression targets b = beta'd + gamma'z, s = -z/gamma
# transform accordingly, so imposing G on the model is free variance reduction.
# The tangent-space field must be: invariant under rotation, ODD under negation,
# and permuted under site reversal. These three are the guard for every later
# phase -- mean subtraction and node features both have to preserve them.


def _sym_model(n_neighbors: int = 2, **kw) -> XYChainGNN:
    """A float64 model, so exact symmetries can be asserted at machine precision."""
    return _make_model(n_neighbors, **kw).double()


def test_rotation_invariance():
    """theta -> theta + phi leaves every wrapped difference, hence v, unchanged."""
    net = _sym_model()
    x, t = _random_angles(4, 8, seed=40).double(), torch.rand(4).double()
    for phi in (0.3, 2.0, -1.7):
        assert torch.allclose(net(t, x + phi), net(t, x), atol=1e-10)


def test_spin_reflection_equivariance():
    """theta -> -theta must flip the field's sign exactly (review F4).

    `trans = d_theta * phi` is odd iff `phi` is EVEN in the configuration. The
    sin(k*d_theta) channels of `edge_attr` are odd, which is what breaks it -- so
    this fails before the cos-only change and holds by construction after.
    """
    net = _sym_model()
    x, t = _random_angles(4, 8, seed=41).double(), torch.rand(4).double()
    v, v_neg = net(t, x), net(t, -x)
    rel = ((v_neg + v).norm() / v.norm()).item()
    assert rel < 1e-10, f"oddness defect = {rel:.3e}"


def test_site_reversal_equivariance():
    """theta_i -> theta_{N+1-i} must permute the field the same way."""
    net = _sym_model()
    x, t = _random_angles(4, 8, seed=42).double(), torch.rand(4).double()
    v_rev, rev_v = net(t, x.flip(-1)), net(t, x).flip(-1)
    rel = ((v_rev - rev_v).norm() / rev_v.norm()).item()
    assert rel < 1e-10, f"reversal defect = {rel:.3e}"


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
        test_cos_expand_values_and_parity,
        test_chain_edge_index_matches_brute_force,
        test_chain_edge_index_is_cached,
        test_batch_graph_is_cached_and_matches_uncached,
        test_chain_graph_no_self_loops_and_in_range,
        test_output_shape_and_finiteness,
        test_single_instance_handles_multiple_chain_lengths,
        test_hidden_and_time_dim_are_decoupled,
        test_mlp_layers_controls_depth,
        test_scalar_time_broadcasts,
        test_invariance_to_global_2pi_shift,
        test_invariance_to_per_node_2pi_shift,
        test_rotation_invariance,
        test_spin_reflection_equivariance,
        test_site_reversal_equivariance,
        test_time_features_are_injective_at_the_endpoints,
        test_network_separates_the_time_endpoints,
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
