from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from PIL import Image

import vascx_models.cli as cli_module


def test_cli_run_loads_config_and_invokes_pipeline(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(input_dir / "sample.png")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("overlay:\n  enabled: true\n", encoding="utf-8")

    captured: dict[str, object] = {}
    fake_app_config = SimpleNamespace(source_path=config_path)

    monkeypatch.setattr(
        "vascx_models.cli.load_app_config", lambda path: fake_app_config
    )
    monkeypatch.setattr(
        "vascx_models.cli.pipeline_ops.run_pipeline",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "run",
            str(input_dir),
            str(output_dir),
            "--config",
            str(config_path),
            "--no-preprocess",
            "--no-vessels",
            "--no-disc",
            "--no-quality",
            "--no-fovea",
            "--no-overlay",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["app_config"] is fake_app_config
    assert captured["data_path"] == str(input_dir)
    assert captured["output_path"] == str(output_dir)
    assert captured["deps"].resolve_device is cli_module.resolve_device


def test_cli_vessel_metrics_loads_config_and_invokes_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("overlay:\n  enabled: true\n", encoding="utf-8")

    captured: dict[str, object] = {}
    fake_app_config = SimpleNamespace(source_path=config_path)

    monkeypatch.setattr(
        "vascx_models.cli.load_app_config", lambda path: fake_app_config
    )
    monkeypatch.setattr(
        "vascx_models.cli.pipeline_ops.run_vessel_metrics_pipeline",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "vessel-metrics",
            str(source_dir),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["app_config"] is fake_app_config
    assert captured["source_output_path"] == source_dir
    assert captured["output_path"] is None
    assert captured["deps"].generate_disc_circles is cli_module.generate_disc_circles


def test_cli_extract_pdf_fundus_invokes_extractor(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    destination_dir = tmp_path / "images"

    captured: dict[str, object] = {}

    def fake_extract_pdf_directory(**kwargs):
        captured.update(kwargs)
        return [destination_dir / "sample.png"]

    monkeypatch.setattr(
        "vascx_models.cli.extract_pdf_directory", fake_extract_pdf_directory
    )

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "extract-pdf-fundus",
            str(source_dir),
            str(destination_dir),
            "--overwrite",
            "--no-crop-margins",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["source_dir"] == source_dir
    assert captured["destination_dir"] == destination_dir
    assert captured["overwrite"] is True
    assert captured["crop_margins"] is False
    assert f"Extracted 1 PDF fundus images into {destination_dir}" in result.output
