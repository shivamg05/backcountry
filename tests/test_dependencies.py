"""What an app's own dependencies must be able to sit beside."""
import tomllib
from pathlib import Path


def test_mlx_is_never_pinned_and_is_required_on_macos_only():
    # backcountry works against the newest MLX: an exact pin kept apps on an
    # old one. uv resolves a project for every platform, and mflux pins Linux's
    # CUDA MLX (mlx[cuda13]==0.32.2), so a requirement on every platform made
    # `uv add backcountry` fail in every mflux app
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    requirement = next(d for d in project["dependencies"] if d.split(";")[0].strip().startswith("mlx"))
    spec, marker = (part.strip() for part in requirement.split(";", 1))
    assert spec.startswith("mlx>=") and "<" not in spec and "==" not in spec, requirement
    assert marker == 'sys_platform == "darwin"', requirement
