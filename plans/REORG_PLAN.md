# Reorganize `eesi/` by system rather than by layer

## Why

The current layout groups by *kind* (`datasets/`, `models/`, `train/`), but the
contents are disjoint by *system*: nothing in `datasets/xy.py` is ever used
together with `models/lj13_dynamics.py`. The grouping therefore costs a
four-directory walk to touch one experiment and buys no reuse.

Worse, it has produced two **backwards dependencies** — the supposedly general
core imports from the system-specific leaves:

```
eesi/interpolant.py:59   from .models.lj13_dynamics import divergence   # core -> LJ13
eesi/ot.py:58            from .models.xygnn import angle_wrap           # core -> XY
```

Regrouping by system turns both into sibling imports inside one folder, and
leaves a core that genuinely depends on nothing but torch.

## Target layout

```
eesi/
  __init__.py            re-exports core + each system's public names
  interpolant.py         EESI base, path/gamma schedules, divergence, losses
  ot.py                  center, _outer_assignment, _hungarian_nd, transport_cost
  data.py                ParticleDataset, make_loader   (was datasets/data.py)
  systems/
    __init__.py
    toy/                 the 1D/2D pedagogical examples
      data.py            GaussianMixture              (was datasets/base.py)
      mlp.py             TimeMLP                      (was models/mlp.py)
    xy/
      data.py            mcxy, sample_p1_exact, energy (was datasets/xy.py)
      gnn.py             XYChainGNN, angle_wrap        (was models/xygnn.py)
      interpolant.py     xyEESI + _min_image           (split from interpolant.py)
      ot.py              xy_ot_couple, xy_cost_matrix, xy_transport_cost
      train.py           (was train/xy.py)
    lj13/
      data.py            lj_energy, sample_prior, ...  (was datasets/lj13.py)
      dynamics.py        LJ13Dynamics, free_energy...  (was models/lj13_dynamics.py)
      interpolant.py     LJ13EESI                      (split from interpolant.py)
      ot.py              equivariant_ot_couple, lj_cost_matrix
      train.py           (was train/lj13.py)
```

Dependency rule after the move: `systems/*` may import from the core and from
its own siblings; the core imports nothing from `systems/`.

## Phases

### Phase 0 — prep
Branch off `main` (`reorg-by-system`), confirm a clean tree. The stale
`build/lib/eesi/` tree is untracked and gitignored; delete it locally so it
stops polluting greps.

### Phase 1 — delete `eesi/utils.py`
Dead in full (see audit below). Remove the file and its README line. Do this
first so the later phases have 192 fewer lines to relocate.

### Phase 2 — whole-file moves (`git mv`, no content edits)
These are pure renames; git records them as such, so history follows.

| from | to |
|---|---|
| `datasets/base.py` | `systems/toy/data.py` |
| `models/mlp.py` | `systems/toy/mlp.py` |
| `datasets/xy.py` | `systems/xy/data.py` |
| `models/xygnn.py` | `systems/xy/gnn.py` |
| `train/xy.py` | `systems/xy/train.py` |
| `datasets/lj13.py` | `systems/lj13/data.py` |
| `models/lj13_dynamics.py` | `systems/lj13/dynamics.py` |
| `train/lj13.py` | `systems/lj13/train.py` |
| `datasets/data.py` | `data.py` |

### Phase 3 — split `interpolant.py` (716 L)
Move `xyEESI` (605–648) plus its helper `_min_image` (224–233) to
`systems/xy/interpolant.py`; move `LJ13EESI` (650–716) to
`systems/lj13/interpolant.py`, where its `divergence` import becomes
`from .dynamics import divergence`. Delete line 59 from the core. Both
subclasses are clean single-hook overrides (`_interpolant_sample`,
`_noise_like`/`_divergence`), so the cut is along an existing seam.

### Phase 4 — split `ot.py` (369 L)
Three ways, along the existing section comments:

- **core** — `center`, `_outer_assignment`, `_hungarian_nd`, `transport_cost`
- **lj13** — `_svdvals_3x3`, `lj_cost_matrix`, `_apply_alignment`, `equivariant_ot_couple`
- **xy** — `_xy_group_elements`, `xy_cost_matrix`, `_xy_apply_alignment`, `xy_ot_couple`, `xy_transport_cost`

`_outer_assignment` is used by both couplers, so it stays core and is imported
by each. `angle_wrap` becomes a sibling import inside `systems/xy/`.
The long module docstring at the head of `ot.py` should be split to follow its
subject matter rather than left wholesale on the core.

### Phase 5 — `__init__.py` files
Delete `datasets/__init__.py`, `models/__init__.py`, `train/__init__.py`; add
one per system package. Keep top-level `eesi/__init__.py` exporting the **same
names as today** — that alone keeps `from eesi import EESI, TimeMLP,
GaussianMixture` working, which is all the 1D/2D notebooks use. Preserve
`train/__init__.py`'s rule that importing `eesi` must not drag in training loops.

### Phase 6 — update callers
- **10 test files** — move into `tests/{core,toy,xy,lj13}/` mirroring the source.
- **2 benchmarks** — `bench_ot.py`, `xy_phase_bench.py`.
- **5 notebooks** — only the import cells change:
  - `1D_NonGaussian`, `2D_GMM` — no change (top-level import survives).
  - `XY_chain_eqOT` — `models.xygnn`/`ot`/`train.xy` → `systems.xy.*`
  - `LJ13_eqOT`, `lj13_sampling` — `datasets.lj13`/`models.lj13_dynamics`/`ot`/`train.lj13` → `systems.lj13.*`

  Edit the import cells only; do not re-execute, so stored outputs and the
  committed figures stay as they are.
- `python -m eesi.train.xy` becomes `python -m eesi.systems.xy.train`; update
  the `train/__init__.py` docstring and README accordingly.

### Phase 7 — verify
1. `pytest` — must pass exactly as before; this is the real check.
2. Import smoke: `python -c "import eesi"` plus one deep import per system.
3. Layering check: `grep -rn "from \.\.\|from eesi\." eesi/*.py` must show no
   hit pointing into `systems/`.
4. Notebook imports only: run the first cell of each of the 5 notebooks.
5. Confirm the checkpoints still load — all are plain `state_dict`s (verified,
   no pickled class paths), so this should be free.

### Phase 8 — README
Redraw the directory tree; drop the `utils.py` line.

## Decisions taken (flag if you disagree)

- **`TimeMLP` → `toy/`.** Its only production use is the 1D/2D notebooks, though
  `tests/test_interpolant.py` also uses it as a generic test net. Alternative is
  keeping it core as `eesi/mlp.py`.
- **`ParticleDataset` → core `eesi/data.py`.** Generic positions+species+box
  plumbing, but note its only current caller is `tests/test_smoke.py` — it is a
  candidate for deletion on the same grounds as `utils.py` if you don't intend
  to use it.
- **Clean break, no compatibility shims** at the old module paths. The repo is
  small and every caller is in-tree.

## As implemented (2026-07-27, branch `reorg-by-system`)

Executed as written, with four deviations:

- **`ParticleDataset`/`make_loader` deleted**, not moved to core — unused outside
  its own smoke test, on the same grounds as `utils.py`. `datasets/data.py` and
  `tests/test_smoke.py::test_dataset_and_loader` are gone.
- **`tests/test_ot.py` → `tests/core/`**, not `tests/lj13/`: it exercises both
  couplings against the one `ot_reference.py` oracle. It now imports the three
  OT modules as `ot` / `lj_ot` / `xy_ot`.
- **`test_eesi.py` kept unique basenames** (`test_xy_eesi.py`,
  `test_lj13_eesi.py`). pytest's default import mode cannot collect two
  same-named test modules in `__init__`-less directories.
- **The OSF checkpoint moved with its module** to
  `eesi/systems/lj13/LJ13_eq_OT_flow_matching`, since `CKPT_PATH` resolves from
  `__file__`. Untracked and gitignored, so it moved outside git.

Notebooks left untouched by request — their imports still point at the old
paths and need updating before they will run.

Result: 133 passed, exactly the 134-test baseline minus the deleted
`ParticleDataset` test.

## Risks

Low. Phases 2–4 are moves, not rewrites; no numerical code changes, so `pytest`
is a sufficient gate. The one fiddly part is notebook JSON — edit import cells
surgically to avoid churning stored outputs.

---

# Appendix: `eesi/utils.py` audit

**Every public symbol is unreferenced outside the file.** It is not imported by
`eesi/__init__.py`, any module, test, benchmark, or notebook.

| symbol | external uses |
|---|---|
| `set_seed` | none |
| `auto_device` | none |
| `EMA` | none |
| `save_ckpt` | none |
| `load_ckpt` | none |
| `eval_hook` | none |
| `_pairwise_distances` / `_pairwise_overlaps` | internal to `eval_hook` only |

The only hit repo-wide is the README line describing the file. The training
modules do their own thing: `train/{xy,lj13}.py` call
`torch.save(model.state_dict(), ...)` directly rather than `save_ckpt`.

It is also **rotted**, not merely unused. `eval_hook` does

```python
_a0, x0 = base.sample((B,))
```

but `GaussianMixture.sample()` returns a single tensor of shape
`sample_shape + (d,)`. That unpack raises `ValueError` for any batch size other
than 2. The helper cannot have run successfully against the current API.

**Recommendation: delete the file.** It is recoverable from git history if EMA
or checkpointing is wanted later, and the version there would need the
`base.sample` fix anyway. `set_seed`/`auto_device` are three lines each to
rewrite at the point of use.
