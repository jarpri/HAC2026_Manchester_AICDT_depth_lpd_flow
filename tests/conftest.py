import sys
from pathlib import Path

# Make the repo root importable, so the tests find hac26 without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
