from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent_runtime_profile"
    / ".claude"
    / "skills"
    / "generate-assets"
    / "scripts"
    / "generate_asset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("test_generate_asset_skill_module", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeProjectManager:
    def __init__(self, project: dict, project_path: Path | None = None):
        self.project = project
        self.project_path = project_path or Path("/tmp/demo")

    @classmethod
    def from_cwd(cls):
        raise AssertionError("from_cwd should be monkeypatched in this test")

    def load_project(self, project_name: str):
        return self.project

    def get_project_path(self, project_name: str):
        return self.project_path


def test_snapshot_current_image_backend_prefers_project_backend():
    module = _load_module()
    pm = _FakeProjectManager({"image_backend": "custom-9/my-image-model"})

    snapshot = module._snapshot_current_image_backend(pm, "demo")

    assert snapshot == {
        "image_provider": "custom-9",
        "image_model": "my-image-model",
    }


def test_snapshot_current_image_backend_falls_back_to_default(monkeypatch):
    module = _load_module()
    pm = _FakeProjectManager({})

    monkeypatch.setattr(module, "_load_default_image_backend", lambda: ("jimeng", "jimeng-4.6"))

    snapshot = module._snapshot_current_image_backend(pm, "demo")

    assert snapshot == {
        "image_provider": "jimeng",
        "image_model": "jimeng-4.6",
    }


def test_generate_single_enqueues_current_image_backend(monkeypatch, tmp_path: Path):
    module = _load_module()
    pm = _FakeProjectManager(
        {"characters": {"Alice": {"description": "一个短发角色"}}},
        project_path=tmp_path,
    )
    captured: dict = {}

    monkeypatch.setattr(module.ProjectManager, "from_cwd", classmethod(lambda cls: (pm, "demo")))
    monkeypatch.setattr(
        module,
        "_snapshot_current_image_backend",
        lambda current_pm, project_name: {"image_provider": "jimeng", "image_model": "jimeng-4.6"},
    )

    def _fake_enqueue_and_wait(**kwargs):
        captured.update(kwargs)
        return {"result": {"file_path": "characters/Alice.png", "version": 3}}

    monkeypatch.setattr(module, "enqueue_and_wait", _fake_enqueue_and_wait)

    output_path = module.generate_single("character", "Alice")

    assert output_path == tmp_path / "characters" / "Alice.png"
    assert captured["payload"] == {
        "prompt": "一个短发角色",
        "image_provider": "jimeng",
        "image_model": "jimeng-4.6",
    }


def test_build_specs_include_backend_snapshot(monkeypatch):
    module = _load_module()
    pm = _FakeProjectManager(
        {
            "props": {
                "玉佩": {"description": "温润透亮的祖传玉佩"},
                "密信": {"description": "折痕明显的旧信件"},
            }
        }
    )

    monkeypatch.setattr(
        module,
        "_snapshot_current_image_backend",
        lambda current_pm, project_name: {"image_provider": "openai", "image_model": "gpt-image-1.5"},
    )

    specs = module._build_specs(pm, "demo", "prop", ["玉佩", "密信"])

    assert [spec.resource_id for spec in specs] == ["玉佩", "密信"]
    assert specs[0].payload == {
        "prompt": "温润透亮的祖传玉佩",
        "image_provider": "openai",
        "image_model": "gpt-image-1.5",
    }
    assert specs[1].payload == {
        "prompt": "折痕明显的旧信件",
        "image_provider": "openai",
        "image_model": "gpt-image-1.5",
    }
