"""Tests for `eesi.systems.tap.ot`: the O(3) equivariant-OT coupling.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_ot.py
    python tests/tap/test_tap_ot.py

Validated against `tests/core/ot_reference.py` (`tap_ot_map`), whose rotation comes
from scipy's `orthogonal_procrustes` rather than from our eigvalsh trick.

Two properties here are not shared with LJ13 and are the reason this module exists
separately rather than as a flag on `equivariant_ot_couple`:

  * nothing is CENTERED -- O(3) acts about the pinned tail, not the centroid;
  * there is no PERMUTATION layer -- the active chain is directed, so relabelling
    monomers is not a symmetry.

Each has a dedicated test below that a naive port of the LJ13 coupling would fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eesi.ot import transport_cost
from eesi.systems.tap.data import bond_vectors, end_to_end_sq, sample_prior
from eesi.systems.tap.ot import _apply_alignment, tap_cost_matrix, tap_ot_couple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ot_reference as ref  # noqa: E402

N, RE_SQR = 8, 4.0


# ---- helpers ---------------------------------------------------------------


def _pair(B: int = 12, seed: int = 0, n: int = N):
    """A (noise, data) pair. The 'data' is a stretched chain, so the two differ."""
    g = torch.Generator().manual_seed(seed)
    x0 = sample_prior(B, RE_SQR, n_particles=n, generator=g)
    x1 = sample_prior(B, RE_SQR * 2.5, n_particles=n, generator=g)
    return x0, x1


def _rot(x: torch.Tensor, seed: int, proper: bool = True) -> torch.Tensor:
    R = torch.tensor(Rotation.random(random_state=seed).as_matrix(), dtype=x.dtype)
    if not proper:
        R = R * torch.tensor([1.0, 1.0, -1.0], dtype=x.dtype)   # det -> -1
    return x @ R.T


def _perm(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute the MONOMER ordering (not the batch)."""
    g = torch.Generator().manual_seed(seed)
    return x[:, torch.randperm(x.shape[1], generator=g)]


def _cost(x0, x1, **kw) -> float:
    return transport_cost(*tap_ot_couple(x0, x1, **kw)).item()


# ---- oracle equivalence ----------------------------------------------------


@pytest.mark.parametrize("proper", [False, True])
def test_cost_matrix_matches_oracle(proper):
    x0, x1 = _pair()
    M = tap_cost_matrix(x0, x1, proper=proper)
    M_ref, _ = ref.tap_cost_matrix(x0, x1, proper=proper)
    assert np.allclose(M.numpy(), M_ref, atol=1e-8), np.abs(M.numpy() - M_ref).max()


@pytest.mark.parametrize("proper", [False, True])
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("batch", [True, False])
def test_all_ablation_cells_match_oracle(proper, align, batch):
    """All four align/batch cells agree with the scipy reference, in both groups."""
    x0, x1 = _pair(seed=1)
    a, b = tap_ot_couple(x0, x1, align=align, batch=batch, proper=proper)
    a_ref, b_ref = ref.tap_ot_map(x0, x1, align=align, batch=batch, proper=proper)
    assert torch.allclose(a, a_ref, atol=1e-8), (a - a_ref).abs().max().item()
    assert torch.allclose(b, b_ref, atol=1e-12)


# ---- the tail stays pinned (the "did someone center it?" guard) ------------


def test_coupling_keeps_the_tail_at_the_origin():
    """The single most important regression guard in this file.

    A port of `eesi.systems.lj13.ot.equivariant_ot_couple` that kept its `center()`
    calls would move the tail off the origin, silently leaving the subspace that the
    prior, the divergence basis and the velocity field are all defined on. Every
    other test here would still pass.
    """
    x0, x1 = _pair(seed=2)
    for align in (True, False):
        for batch in (True, False):
            a, b = tap_ot_couple(x0, x1, align=align, batch=batch)
            assert a[:, 0].abs().max() < 1e-12, (align, batch, "noise tail moved")
            assert b[:, 0].abs().max() < 1e-12, (align, batch, "data tail moved")


def test_alignment_is_a_rotation_about_the_origin():
    """Alignment preserves every distance to the origin, particle by particle.

    A rotation about the centroid would not: it would change |x_i| for each i.
    """
    x0, x1 = _pair(seed=3)
    a = _apply_alignment(x0, x1)
    assert torch.allclose(a.norm(dim=-1), x0.norm(dim=-1), atol=1e-10)


def test_coupling_preserves_internal_geometry():
    """Bond lengths are untouched: the group acts rigidly, it does not deform."""
    x0, x1 = _pair(seed=4)
    a, _ = tap_ot_couple(x0, x1)
    before = sorted(bond_vectors(x0).norm(dim=-1).flatten().tolist())
    after = sorted(bond_vectors(a).norm(dim=-1).flatten().tolist())
    assert np.allclose(before, after, atol=1e-10)


# ---- no permutation layer (the physics of a directed chain) ----------------


def test_permuting_monomers_of_x1_changes_the_cost():
    """Relabelling monomers is NOT a symmetry -- the head-to-tail asymmetry is real.

    This is the direct encoding of why TAP needs its own coupling. Under LJ13's
    S(N) x SO(3) coupling a permuted copy of a configuration is free (cost ~ 0);
    under TAP's O(3)-only coupling it is expensive. A copy-paste of the LJ13
    Hungarian into this module would flip the assertion.
    """
    from eesi.systems.lj13.ot import equivariant_ot_couple

    _, x1 = _pair(B=6, seed=5)
    x0 = _perm(x1, seed=99)                    # same cloud, monomers relabelled
    assert not torch.allclose(x0, x1)          # the permutation is nontrivial

    tap_cost = transport_cost(*tap_ot_couple(x0, x1, batch=False)).item()
    lj_cost = transport_cost(*equivariant_ot_couple(x0, x1, batch=False)).item()

    scale = (x1 ** 2).sum(dim=(-1, -2)).mean().item()
    assert lj_cost < 1e-8 * max(scale, 1.0), f"LJ13 should absorb a permutation: {lj_cost}"
    assert tap_cost > 0.05 * scale, f"TAP must NOT absorb a permutation: {tap_cost}"


# ---- invariance ------------------------------------------------------------


def test_invariant_to_rotating_the_noise():
    """Rotating x0 before coupling leaves the cost unchanged: the group is quotiented out."""
    x0, x1 = _pair(seed=6)
    assert abs(_cost(x0, x1) - _cost(_rot(x0, 1), x1)) < 1e-8


def test_rotating_x1_alone_is_EXACTLY_invariant_unlike_lj13():
    """The alignment is exact here, so rotating the DATA alone changes nothing either.

    LJ13 cannot say this (see tests/core/test_ot.py::test_rotating_x1_alone_is_not_
    invariant): its Hungarian runs on unrotated coordinates, so the sequential
    permutation-then-rotation approximation depends on x1's orientation. With no
    permutation stage, orthogonal Procrustes solves the whole group exactly and the
    cost matrix is a genuine invariant of the pair of orbits.
    """
    x0, x1 = _pair(seed=7)
    M = tap_cost_matrix(x0, x1)
    for s in (1, 2, 3):
        assert torch.allclose(tap_cost_matrix(x0, _rot(x1, s)), M, atol=1e-8)


def test_invariant_to_reflecting_either_side_under_o3():
    """With proper=False an improper transform is free on either argument."""
    x0, x1 = _pair(seed=8)
    M = tap_cost_matrix(x0, x1, proper=False)
    assert torch.allclose(tap_cost_matrix(-x0, x1, proper=False), M, atol=1e-8)
    assert torch.allclose(tap_cost_matrix(x0, -x1, proper=False), M, atol=1e-8)


# ---- O(3) vs SO(3) ---------------------------------------------------------


def test_reflection_branch_is_live():
    """A mirrored pair is free under O(3) and costs real money under SO(3).

    If `proper` were ignored, or the determinant correction dropped, the two numbers
    below would coincide.

    The SO(3) residue is a few percent of the scale rather than order-one: the best
    proper rotation still aligns two of the three principal axes of a mirrored cloud
    and only pays for flipping the third, so the penalty is ~4 s_3, not ~4 sum(s).
    The assertion is therefore "clearly nonzero", not "comparable to the scale".
    """
    _, x1 = _pair(B=6, seed=9)
    x0 = _rot(x1, seed=4, proper=False)                 # x1 up to a mirror
    c_o3 = transport_cost(*tap_ot_couple(x0, x1, batch=False, proper=False)).item()
    c_so3 = transport_cost(*tap_ot_couple(x0, x1, batch=False, proper=True)).item()
    scale = (x1 ** 2).sum(dim=(-1, -2)).mean().item()
    assert c_o3 < 1e-8 * scale, f"O(3) should absorb the mirror: {c_o3}"
    assert c_so3 > 0.01 * scale, f"SO(3) must not: {c_so3}"
    assert c_so3 > 1e6 * max(c_o3, 1e-18), (c_so3, c_o3)


def test_so3_cost_is_never_below_o3_cost():
    """SO(3) optimizes over a subgroup, so its cost can only be higher."""
    x0, x1 = _pair(seed=10)
    assert (tap_cost_matrix(x0, x1, proper=True)
            >= tap_cost_matrix(x0, x1, proper=False) - 1e-10).all()


# ---- cost properties -------------------------------------------------------


def test_cost_floor_and_self_pairing():
    """Costs are non-negative, and a configuration against itself is free."""
    x0, x1 = _pair(seed=11)
    assert (tap_cost_matrix(x0, x1) > -1e-9).all()
    d = tap_cost_matrix(x1, x1).diagonal()
    assert d.abs().max() < 1e-8, d.abs().max().item()


def test_alignment_achieves_the_cost_matrix_entry():
    """The rotation applied to survivors realizes the cost the matrix advertised.

    Computing a cost with alignment and then interpolating the unaligned pair is the
    classic bug in this family of couplings; this pins the two together.
    """
    x0, x1 = _pair(seed=12)
    M = tap_cost_matrix(x0, x1)
    a, b = tap_ot_couple(x0, x1, batch=False)          # identity pairing
    realized = ((a - b) ** 2).sum(dim=(-1, -2))
    assert torch.allclose(realized, M.diagonal(), atol=1e-8)


def test_full_coupling_beats_every_ablation():
    """align+batch <= each single layer <= neither. The 2x2 ordering."""
    x0, x1 = _pair(B=16, seed=13)
    both = _cost(x0, x1, align=True, batch=True)
    only_a = _cost(x0, x1, align=True, batch=False)
    only_b = _cost(x0, x1, align=False, batch=True)
    neither = _cost(x0, x1, align=False, batch=False)
    assert both <= only_a + 1e-9 and both <= only_b + 1e-9
    assert only_a <= neither + 1e-9 and only_b <= neither + 1e-9


def test_batch_assignment_is_optimal():
    """No permutation of the batch beats the one the coupling chose."""
    x0, x1 = _pair(B=7, seed=14)
    best = _cost(x0, x1, align=True, batch=True)
    g = torch.Generator().manual_seed(0)
    for _ in range(30):
        p = torch.randperm(x1.shape[0], generator=g)
        assert _cost(x0[p], x1, align=True, batch=False) > best - 1e-9


# ---- plumbing --------------------------------------------------------------


def test_no_gradient_flows_through_the_coupling():
    x0, x1 = _pair(seed=15)
    x0 = x0.requires_grad_(True)
    a, b = tap_ot_couple(x0, x1)
    assert not a.requires_grad and not b.requires_grad


def test_float32_is_not_upcast():
    x0, x1 = _pair(seed=16)
    a, b = tap_ot_couple(x0.float(), x1.float())
    assert a.dtype == torch.float32 and b.dtype == torch.float32


def test_mismatched_batch_sizes_are_rejected():
    """A rectangular cost matrix would give a partial assignment; fail loudly."""
    x0, _ = _pair(B=8, seed=17)
    _, x1 = _pair(B=6, seed=18)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        tap_ot_couple(x0, x1)


# ---- marginal preservation -------------------------------------------------


def test_coupled_noise_is_still_a_valid_prior_sample():
    """The aligned noise still looks like an ideal chain, so p0 is preserved.

    The coupling transforms x0 by a group element chosen using x1. That is only
    marginal-preserving because p0 AND p1 are O(3)-invariant; the check that it
    actually worked is that the standard prior statistics survive.
    """
    B = 4000
    g = torch.Generator().manual_seed(19)
    x0 = sample_prior(B, RE_SQR, n_particles=N, generator=g)
    x1 = sample_prior(B, RE_SQR * 2.0, n_particles=N, generator=g)   # O(3)-invariant p1
    a, _ = tap_ot_couple(x0, x1)

    assert abs(end_to_end_sq(a).mean().item() - RE_SQR) / RE_SQR < 0.05
    b_before = (bond_vectors(x0) ** 2).sum(-1).mean().item()
    b_after = (bond_vectors(a) ** 2).sum(-1).mean().item()
    assert abs(b_after - b_before) / b_before < 0.02
    # isotropy: no lab-frame direction should be singled out by the alignment
    var = a.reshape(-1, 3).var(0)
    assert torch.allclose(var, var.mean().expand(3), rtol=0.1), var.tolist()


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_coupling_keeps_the_tail_at_the_origin,
        test_alignment_is_a_rotation_about_the_origin,
        test_coupling_preserves_internal_geometry,
        test_permuting_monomers_of_x1_changes_the_cost,
        test_invariant_to_rotating_the_noise,
        test_rotating_x1_alone_is_EXACTLY_invariant_unlike_lj13,
        test_invariant_to_reflecting_either_side_under_o3,
        test_reflection_branch_is_live,
        test_so3_cost_is_never_below_o3_cost,
        test_cost_floor_and_self_pairing,
        test_alignment_achieves_the_cost_matrix_entry,
        test_full_coupling_beats_every_ablation,
        test_batch_assignment_is_optimal,
        test_no_gradient_flows_through_the_coupling,
        test_float32_is_not_upcast,
        test_mismatched_batch_sizes_are_rejected,
        test_coupled_noise_is_still_a_valid_prior_sample,
    ]
    for _p in (False, True):
        tests.append(lambda p=_p: test_cost_matrix_matches_oracle(p))
        for _a in (True, False):
            for _b in (True, False):
                tests.append(lambda p=_p, a=_a, b=_b:
                             test_all_ablation_cells_match_oracle(p, a, b))
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
