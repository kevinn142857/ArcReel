"""CLI bootstrap tests for generate-script skill scripts.

Ensure the skill scripts can import ``lib`` even when launched via plain
``python`` from arbitrary working directories.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = [
    REPO_ROOT / "agent_runtime_profile" / ".claude" / "skills" / "generate-script" / "scripts" / "generate_script.py",
    REPO_ROOT / "agent_runtime_profile" / ".claude" / "skills" / "generate-script" / "scripts" / "normalize_drama_script.py",
]


@pytest.mark.parametrize("script_path", SKILL_SCRIPTS, ids=lambda p: p.stem)
def test_skill_script_bootstraps_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script_path: Path) -> None:
    """Scripts should self-bootstrap repo root into sys.path."""
    for key in [name for name in sys.modules if name == "lib" or name.startswith("lib.")]:
        monkeypatch.delitem(sys.modules, key, raising=False)

    current_sys_path = [entry for entry in sys.path if Path(entry or os.curdir).resolve() != REPO_ROOT]
    monkeypatch.setattr(sys, "path", current_sys_path[:])
    monkeypatch.chdir(tmp_path)

    spec = importlib.util.spec_from_file_location(f"{script_path.stem}_bootstrap_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert str(REPO_ROOT) in sys.path
