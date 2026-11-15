"""tests/fixtures/engine_held_model.py without release(): the model stays held."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo, where `tests` is: a model file finds its own imports

from tests.fixtures.engine_kit_model import build as _build, run  # noqa: F401

HELD = []


def build():
    model = _build()
    HELD.append(model)
    return model
