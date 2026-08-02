"""Put src/ on sys.path so tests can import project modules directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
