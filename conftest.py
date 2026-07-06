"""Pytest config: put the package root on sys.path so `import eesi` works
without `pip install -e .`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
