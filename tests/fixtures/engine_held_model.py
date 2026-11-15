"""tests/fixtures/engine_kit_model.py with the model also kept where dropping
it does not free it, as a running engine thread keeps LM Studio's kit; the
module's release() lets go of every one it built, as an engine's shutdown() does."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo, where `tests` is: a model file finds its own imports

from tests.fixtures.engine_kit_model import build as _build, run  # noqa: F401

HELD = []


def build():
    model = _build()
    HELD.append(model)
    return model


def release():
    HELD.clear()
