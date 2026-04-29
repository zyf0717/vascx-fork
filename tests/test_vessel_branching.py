from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vascx_models.config import OverlayCircle, VesselBranchingConfig
from vascx_models.metrics.vessel_branching import (
    VESSEL_BRANCHING_COLUMNS,
    VESSEL_BRANCHING_WIDTH_COLUMNS,
    measure_vessel_branching_between_disc_circle_pair,
)


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_measure_vessel_branching_writes_branch_and_width_audit_tables(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    height = width = 180
    vessel = np.zeros((height, width), dtype=np.uint8)
    av = np.zeros((height, width), dtype=np.uint8)

    vessel[110:136, 90] = 1
    av[110:136, 90] = 2
    vessel[135, 35:146] = 1
    av[135, 35:146] = 2

    _write_mask(vessels_dir / "sample.png", vessel)
    _write_mask(av_dir / "sample.png", av)

    geometry_path = tmp_path / "disc_geometry.csv"
    pd.DataFrame(
        {
            "x_disc_center": [90.0],
            "y_disc_center": [90.0],
            "disc_radius_px": [20.0],
        },
        index=["sample"],
    ).to_csv(geometry_path)

    branching_output_path = tmp_path / "vessel_branching.csv"
    widths_output_path = tmp_path / "vessel_branching_widths.csv"
    df_branching, df_widths = measure_vessel_branching_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="1r", diameter=1.0),
        outer_circle=OverlayCircle(name="3.5r", diameter=3.5),
        output_path=branching_output_path,
        widths_output_path=widths_output_path,
        branching_config=VesselBranchingConfig(
            inner_circle="1r",
            outer_circle="3.5r",
            min_branch_length_px=5.0,
            width_skip_px=2.0,
            width_sample_length_px=6.0,
            width_samples_per_branch=2,
            angle_sample_px=8.0,
        ),
    )

    assert branching_output_path.exists()
    assert widths_output_path.exists()
    assert df_branching.columns.tolist() == VESSEL_BRANCHING_COLUMNS
    assert df_widths.columns.tolist() == VESSEL_BRANCHING_WIDTH_COLUMNS
    assert len(df_branching) == 1
    assert len(df_widths) == 6

    record = df_branching.iloc[0]
    assert record["image_id"] == "sample"
    assert record["vessel_type"] == "vein"
    assert record["x_junction"] == pytest.approx(90.0)
    assert record["y_junction"] == pytest.approx(135.0)
    assert record["parent_width_px"] == pytest.approx(1.0)
    assert record["daughter_1_width_px"] == pytest.approx(1.0)
    assert record["daughter_2_width_px"] == pytest.approx(1.0)
    assert record["branching_angle_deg"] == pytest.approx(180.0)
    angle_points = sorted(
        [
            (record["daughter_1_angle_x"], record["daughter_1_angle_y"]),
            (record["daughter_2_angle_x"], record["daughter_2_angle_y"]),
        ]
    )
    assert angle_points == pytest.approx([(82.0, 135.0), (98.0, 135.0)])
    assert record["branching_coefficient"] == pytest.approx(2.0)
    assert record["n_parent_width_samples"] == 2
    assert record["n_daughter_1_width_samples"] == 2
    assert record["n_daughter_2_width_samples"] == 2

    assert sorted(df_widths["branch_role"].unique().tolist()) == [
        "daughter_1",
        "daughter_2",
        "parent",
    ]
    assert df_widths["measurement_valid"].tolist() == [True] * 6


def test_measure_vessel_branching_writes_empty_tables_when_no_bifurcation(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    vessel = np.zeros((96, 96), dtype=np.uint8)
    av = np.zeros((96, 96), dtype=np.uint8)
    vessel[20:80, 48] = 1
    av[20:80, 48] = 1
    _write_mask(vessels_dir / "sample.png", vessel)
    _write_mask(av_dir / "sample.png", av)

    geometry_path = tmp_path / "disc_geometry.csv"
    pd.DataFrame(
        {
            "x_disc_center": [48.0],
            "y_disc_center": [48.0],
            "disc_radius_px": [10.0],
        },
        index=["sample"],
    ).to_csv(geometry_path)

    df_branching, df_widths = measure_vessel_branching_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="1r", diameter=1.0),
        outer_circle=OverlayCircle(name="3r", diameter=3.0),
    )

    assert df_branching.empty
    assert df_branching.columns.tolist() == VESSEL_BRANCHING_COLUMNS
    assert df_widths.empty
    assert df_widths.columns.tolist() == VESSEL_BRANCHING_WIDTH_COLUMNS
