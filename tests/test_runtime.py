import os
from pathlib import Path

from vascx_models import runtime


def test_configure_runtime_environment_sets_paths_and_links_models(
    tmp_path: Path, monkeypatch
) -> None:
    package_dir = tmp_path / "vascx_models"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    weights_dir = tmp_path / "disc"
    weights_dir.mkdir()
    model_path = weights_dir / "disc_test.pt"
    model_path.write_text("model", encoding="utf-8")

    monkeypatch.setattr(runtime, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("RTNLS_MODEL_RELEASES", raising=False)

    runtime.configure_runtime_environment()

    assert Path(os.environ["MPLCONFIGDIR"]) == tmp_path / ".mplconfig"
    assert Path(os.environ["XDG_CACHE_HOME"]) == tmp_path / ".cache"
    assert Path(os.environ["RTNLS_MODEL_RELEASES"]) == tmp_path / "model_releases"

    linked_model = tmp_path / "model_releases" / "disc_test.pt"
    assert linked_model.is_symlink()
    assert linked_model.resolve() == model_path.resolve()
