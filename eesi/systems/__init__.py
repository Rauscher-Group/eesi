"""Concrete systems, one subpackage each.

Each subpackage is self-contained -- data, network, interpolant specialisation,
OT coupling, and training entry point live together:

    eesi.systems.gmm     1D/2D Gaussian-mixture examples
    eesi.systems.xy      the 1D XY chain
    eesi.systems.lj13    the LJ13 cluster
    eesi.systems.tap     the tangentially active polymer

Dependency rule: a system may import from the core (`eesi.interpolant`,
`eesi.ot`, `eesi.egnn`) and from its own siblings, never from another system and
never back into the core. When two systems need the same component, the component
moves into the core rather than one system importing the other: that is why the
E(n)-GNN backbone lives in `eesi.egnn` and not in `eesi.systems.lj13`, even though
LJ13 was the only user for a while.

Each system's `train` module exposes a `train(...)` callable plus an argparse
`main()`, so it can be driven either from a notebook in `experiments/` or from
the command line:

    python -m eesi.systems.xy.train --steps 2000 --batch 256 --J 1.0
    python -m eesi.systems.lj13.train --steps 2000 --batch 64

TAP additionally has a `run` module, because its runs are long enough that a flag
list and a notebook cell stop being enough. It reads a YAML config, checkpoints as
it goes, and can be resumed after a crash; it also carries the sampling and entropy
jobs, so the notebook is left doing analysis and figures:

    python -m eesi.systems.tap.run train --config experiments/TAP/configs/tap_N20_Pe0.yaml

It is an orchestration layer over `tap.train.train_si`, not a second training loop.
The generic parts -- the YAML schema vocabulary, the run directory, the checkpoint
format -- live in `eesi.config` and `eesi.rundir` so another system can adopt them
without importing anything from `tap`.

Deliberately not imported by any package `__init__`: importing the package should
not drag in the training loops.
"""
