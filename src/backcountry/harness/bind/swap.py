"""Harness-side swap: deliberately thin over backcountry_runtime.swap, so the
harness installs wrappers with exactly the code the artifact ships."""

from backcountry_runtime.swap import ReplayWrapper, install, resolve, uninstall

__all__ = ["ReplayWrapper", "install", "resolve", "uninstall"]
