# A config-driven entry point for TAP runs

## Why

TAP training outgrew the notebook. The other systems got away with a few thousand
steps, which a cell can hold; TAP wants 100k per learning-rate stage and more. In
`TAP.ipynb` the workflow was cell 7 for prior calibration, cell 12 for the model, cell
13 for the checkpoint load, and cell 14 for a `train_si` call that was re-executed by
hand with a smaller `lr` to stage the schedule. Three things were wrong with that at
this length:

* A kernel restart lost the run. There was no checkpoint between the start and the end
  of a cell.
* Re-running the calibration cell against a differently-sliced dataset silently
  produced a prior the network had never been trained against. Nothing said so, and
  every entropy number downstream would have been quietly wrong.
* The older `python -m eesi.systems.tap.train` CLI never exposed `hidden_nf`,
  `n_layers`, `path`, `gamma_scale` or `eps`, and had no resume at all.

The shape of the fix: a run is described by one YAML file and lives in one directory.
Long jobs are launched detached under tmux, checkpoint as they go, and resume exactly
where they stopped. The notebook is left with analysis and figures.

## What was built

Generic, in the core, because none of it knows about polymers:

* `eesi/config.py` -- YAML loading, `--set` overrides, and the validator vocabulary
  (`_closed`, `_typed`, `_choice`, ...) that each system's schema is assembled from.
  Plus `StageConfig`/`CheckpointConfig`, which are the same everywhere.
* `eesi/rundir.py` -- the run directory, atomic checkpoints, the append-only history
  file, `split_nets`, and the RNG capture/restore that resume depends on.

TAP-specific:

* `eesi/systems/tap/config.py` -- the schema, and `resolve_prior`, which turns the
  three routes to a prior parameter (given, measured, fitted) into four numbers plus
  a provenance string for each.
* `eesi/systems/tap/run.py` -- `train`, `sample`, `entropy`, `export`, `status`, and
  `load_run`, which is what the notebook imports.
* `experiments/TAP/configs/tap_N20_Pe0.yaml` -- the notebook's settings, commented.

`eesi/systems/tap/train.py` gained four optional keyword arguments (`opt`, `callback`,
`start_step`, and `seed=None`) and nothing else. That was the alternative to copying
the loop into the runner, where the copy would have drifted. With all four at their
defaults the loop is what it always was, and a test pins that.

## Decisions worth remembering

**A resume reads its config from the checkpoint, not from the YAML file.** By the time
anyone comes back to a stopped run the file has usually been edited. `--set` is refused
on resume for the same reason.

**The checkpoint's prior always wins on resume**, with a warning when the config would
now resolve differently and `--strict-prior` to make that fatal. Re-measuring would put
a different base distribution under a half-trained flow.

**History is CSV, not `.npy`.** It can be appended to without a rewrite, truncated back
to a checkpoint's step count, and followed with `tail -f` while the job runs. Rows are
flushed *before* the checkpoint that counts them: a crash between the two leaves extra
rows, which the resume drops. The other order loses rows, which it cannot.

**Checkpoints are written to a temporary file and `os.replace`d.** A crash mid-write
leaves the previous checkpoint intact instead of a truncated file that loads as
garbage.

**`k` and `b` default to the simulation Hamiltonian's values (100.0, 1.0)**, not the
data's moments (~91.5, ~1.024). The thermodynamic-integration bridge subtracts an
analytic prior-vs-ideal offset, and that is only valid if the prior's bond law is the
simulation's. The `measure` route stays available for when no reconciliation is being
done.

**Both checkpoint formats stay reachable.** The run's own checkpoint is a superset of
what `train.py --out` writes, and `split_nets` derives the bare per-network files the
notebook has always loaded, written into `nets/` on every checkpoint.

## Two things a future reader will want to know

`gamma = 1/(2 var(cos theta))` is a **definition**, not an inverse: data generated at
gamma=20 puts it at about 38, because the bending weight lives on the sphere and
`var(cos theta)` picks up the `sin theta` measure and the truncation to [-1, 1].
`cos_theta_0` is then fitted to fix the resulting chain size. `tests/tap/
test_tap_config.py` has a test whose only job is to stop someone "correcting" this,
which would silently change every prior the runner builds.

`k = 1/var(Q)` and `b = E[Q]` *do* invert the bond law, but only when the chain is
stiff: within about 2% at the physical k=100, off by 21% at k=8. The q^2 Jacobian is
what does it.

## Verified

A 200-step run on `tap_N20_Pe0.npy`, done twice: straight through, and interrupted with
a real SIGINT at step 37 of stage 1 then resumed. The weights come out **bitwise
identical** (maximum absolute difference exactly 0.0), and the history has every step
once, 0 through 199. `resolve_prior` on the full 100k dataset reproduces cell 7 exactly:
`k=100.0000  b=1.0000  gamma=2.3309  cos_theta_0=0.5346`, `S[p0]=+25.1483` nats.

## Not done here

The notebook itself still has its training and calibration cells. Rewriting them to
read from a run directory is the remaining piece, and wants a real long run to have
finished first so the committed outputs are genuine. The Rouse diagnostics and the TI
bridge stay in the notebook by choice -- they are analysis, not jobs.
