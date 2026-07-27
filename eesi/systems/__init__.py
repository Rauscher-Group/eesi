"""Concrete systems, one subpackage each.

Each subpackage is self-contained -- data, network, interpolant specialisation,
OT coupling, and training entry point live together:

    eesi.systems.toy     1D/2D Gaussian-mixture examples
    eesi.systems.xy      the 1D XY chain
    eesi.systems.lj13    the LJ13 cluster

Dependency rule: a system may import from the core (`eesi.interpolant`,
`eesi.ot`) and from its own siblings, never from another system and never back
into the core.

Each system's `train` module exposes a `train(...)` callable plus an argparse
`main()`, so it can be driven either from a notebook in `experiments/` or from
the command line:

    python -m eesi.systems.xy.train --steps 2000 --batch 256 --J 1.0
    python -m eesi.systems.lj13.train --steps 2000 --batch 64

Deliberately not imported by any package `__init__`: importing the package should
not drag in the training loops.
"""
