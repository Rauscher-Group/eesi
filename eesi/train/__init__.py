"""Training entry points, one module per system.

Each module exposes a `train(...)` callable plus an argparse `main()`, so it can be
driven either from a notebook in `experiments/` or from the command line:

    python -m eesi.train.xy --steps 2000 --batch 256 --J 1.0
    python -m eesi.train.lj13 --steps 2000 --batch 64

Deliberately not imported by `eesi/__init__.py`: importing the package should not drag
in the training loops.
"""
