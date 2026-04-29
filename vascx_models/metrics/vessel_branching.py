from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..config import OverlayCircle, VesselBranchingConfig
from ..geometry.vessel_masks import typed_vessel_masks
from ..geometry.vessel_paths import (
    VesselBranchingPoint,
    interpolate_path_point,
    path_cumulative_lengths,
    trace_vessel_branching_points_between_disc_circle_pair,
)
from .vessel_widths import _estimate_path_tangent, _measure_mask_width_from_tangent

VESSEL_BRANCHING_COLUMNS = [
    "image_id",
    "inner_circle",
    "outer_circle",
    "inner_circle_radius_px",
    "outer_circle_radius_px",
    "connection_index",
    "x_junction",
    "y_junction",
    "parent_width_px",
    "daughter_1_width_px",
    "daughter_2_width_px",
    "branching_angle_deg",
    "daughter_1_angle_x",
    "daughter_1_angle_y",
    "daughter_2_angle_x",
    "daughter_2_angle_y",
    "branching_coefficient",
    "vessel_type",
    "parent_path_length_px",
    "daughter_1_path_length_px",
    "daughter_2_path_length_px",
    "n_parent_width_samples",
    "n_daughter_1_width_samples",
    "n_daughter_2_width_samples",
]

VESSEL_BRANCHING_WIDTH_COLUMNS = [
    "image_id",
    "inner_circle",
    "outer_circle",
    "inner_circle_radius_px",
    "outer_circle_radius_px",
    "connection_index",
    "branch_role",
    "sample_index",
    "x_junction",
    "y_junction",
    "x",
    "y",
    "width_px",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "normal_x",
    "normal_y",
    "vessel_type",
    "measurement_valid",
    "measurement_failure_reason",
]


def _unit(vector_xy: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector_xy))
    if norm == 0.0:
        return np.full_like(vector_xy, float("nan"), dtype=float)
    return vector_xy / norm


def _angle_between(v1_xy: np.ndarray, v2_xy: np.ndarray) -> float:
    v1_xy = _unit(v1_xy)
    v2_xy = _unit(v2_xy)
    if not np.all(np.isfinite(v1_xy)) or not np.all(np.isfinite(v2_xy)):
        return float("nan")
    cosine = float(np.clip(np.dot(v1_xy, v2_xy), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _branch_vector(path_xy: np.ndarray, angle_sample_px: float) -> np.ndarray:
    sample_xy = _branch_angle_sample_point(path_xy, angle_sample_px)
    if not np.all(np.isfinite(sample_xy)):
        return np.array([float("nan"), float("nan")], dtype=float)
    return sample_xy - path_xy[0]


def _branch_angle_sample_point(path_xy: np.ndarray, angle_sample_px: float) -> np.ndarray:
    cumulative_lengths = path_cumulative_lengths(path_xy)
    if len(cumulative_lengths) < 2 or cumulative_lengths[-1] <= 0.0:
        return np.array([float("nan"), float("nan")], dtype=float)
    sample_length = min(float(angle_sample_px), float(cumulative_lengths[-1]))
    return interpolate_path_point(path_xy, cumulative_lengths, sample_length)


def _sample_targets_for_branch(
    cumulative_lengths: np.ndarray,
    config: VesselBranchingConfig,
) -> np.ndarray:
    total_length = float(cumulative_lengths[-1])
    if total_length < config.min_branch_length_px:
        return np.empty(0, dtype=float)

    start = min(float(config.width_skip_px), max(total_length - 1e-6, 0.0))
    end = min(start + float(config.width_sample_length_px), total_length - 1e-6)
    if end <= start:
        return np.asarray([start], dtype=float)

    return np.linspace(start, end, config.width_samples_per_branch, dtype=float)


def _branch_width_sample_records(
    image_id: str,
    vessel_type: str,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    inner_radius_px: float,
    outer_radius_px: float,
    branching_point: VesselBranchingPoint,
    branch_role: str,
    path_xy: np.ndarray,
    vessel_mask: np.ndarray,
    config: VesselBranchingConfig,
) -> list[dict[str, object]]:
    cumulative_lengths = path_cumulative_lengths(path_xy)
    if len(cumulative_lengths) < 2 or cumulative_lengths[-1] <= 0.0:
        return []

    targets = _sample_targets_for_branch(cumulative_lengths, config)
    records: list[dict[str, object]] = []
    tangent_lag_px = max(2.0, config.width_sample_length_px / 2.0)
    for sample_index, target_length in enumerate(targets, start=1):
        center_xy = interpolate_path_point(path_xy, cumulative_lengths, target_length)
        tangent_xy = _estimate_path_tangent(
            path_xy,
            cumulative_lengths,
            float(target_length),
            lag_px=tangent_lag_px,
        )
        if tangent_xy is None:
            measurement = {
                "width_px": float("nan"),
                "x_start": float("nan"),
                "y_start": float("nan"),
                "x_end": float("nan"),
                "y_end": float("nan"),
                "normal_x": float("nan"),
                "normal_y": float("nan"),
                "measurement_valid": False,
                "measurement_failure_reason": "path_tangent_unavailable",
            }
        else:
            measurement = _measure_mask_width_from_tangent(
                vessel_mask=vessel_mask,
                center_xy=center_xy,
                tangent_xy=tangent_xy,
                step_px=config.measurement_step_px,
                refinement_steps=config.boundary_refinement_steps,
                trace_padding_px=config.trace_padding_px,
            )

        records.append(
            {
                "image_id": image_id,
                "inner_circle": inner_circle.name,
                "outer_circle": outer_circle.name,
                "inner_circle_radius_px": inner_radius_px,
                "outer_circle_radius_px": outer_radius_px,
                "connection_index": branching_point.connection_index,
                "branch_role": branch_role,
                "sample_index": sample_index,
                "x_junction": float(branching_point.junction_xy[0]),
                "y_junction": float(branching_point.junction_xy[1]),
                "x": float(center_xy[0]),
                "y": float(center_xy[1]),
                "width_px": float(measurement["width_px"]),
                "x_start": float(measurement["x_start"]),
                "y_start": float(measurement["y_start"]),
                "x_end": float(measurement["x_end"]),
                "y_end": float(measurement["y_end"]),
                "normal_x": float(measurement["normal_x"]),
                "normal_y": float(measurement["normal_y"]),
                "vessel_type": vessel_type,
                "measurement_valid": bool(measurement["measurement_valid"]),
                "measurement_failure_reason": measurement[
                    "measurement_failure_reason"
                ],
            }
        )
    return records


def _median_valid_width(records: list[dict[str, object]]) -> tuple[float, int]:
    widths = [
        float(record["width_px"])
        for record in records
        if bool(record["measurement_valid"]) and np.isfinite(float(record["width_px"]))
    ]
    if not widths:
        return float("nan"), 0
    return float(np.median(widths)), len(widths)


def _branching_record(
    image_id: str,
    vessel_type: str,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    inner_radius_px: float,
    outer_radius_px: float,
    branching_point: VesselBranchingPoint,
    parent_records: list[dict[str, object]],
    daughter_1_records: list[dict[str, object]],
    daughter_2_records: list[dict[str, object]],
    config: VesselBranchingConfig,
) -> dict[str, object]:
    parent_width, n_parent = _median_valid_width(parent_records)
    daughter_1_width, n_daughter_1 = _median_valid_width(daughter_1_records)
    daughter_2_width, n_daughter_2 = _median_valid_width(daughter_2_records)

    daughter_1_vector = _branch_vector(
        branching_point.daughter_paths_xy[0],
        angle_sample_px=config.angle_sample_px,
    )
    daughter_2_vector = _branch_vector(
        branching_point.daughter_paths_xy[1],
        angle_sample_px=config.angle_sample_px,
    )
    daughter_1_angle_xy = _branch_angle_sample_point(
        branching_point.daughter_paths_xy[0],
        angle_sample_px=config.angle_sample_px,
    )
    daughter_2_angle_xy = _branch_angle_sample_point(
        branching_point.daughter_paths_xy[1],
        angle_sample_px=config.angle_sample_px,
    )
    branching_angle = _angle_between(daughter_1_vector, daughter_2_vector)

    if not np.isfinite(parent_width) or parent_width <= 0.0:
        branching_coefficient = float("nan")
    else:
        branching_coefficient = float(
            (daughter_1_width**2 + daughter_2_width**2) / parent_width**2
        )

    parent_length = float(path_cumulative_lengths(branching_point.parent_path_xy)[-1])
    daughter_1_length = float(
        path_cumulative_lengths(branching_point.daughter_paths_xy[0])[-1]
    )
    daughter_2_length = float(
        path_cumulative_lengths(branching_point.daughter_paths_xy[1])[-1]
    )

    return {
        "image_id": image_id,
        "inner_circle": inner_circle.name,
        "outer_circle": outer_circle.name,
        "inner_circle_radius_px": inner_radius_px,
        "outer_circle_radius_px": outer_radius_px,
        "connection_index": branching_point.connection_index,
        "x_junction": float(branching_point.junction_xy[0]),
        "y_junction": float(branching_point.junction_xy[1]),
        "parent_width_px": parent_width,
        "daughter_1_width_px": daughter_1_width,
        "daughter_2_width_px": daughter_2_width,
        "branching_angle_deg": branching_angle,
        "daughter_1_angle_x": float(daughter_1_angle_xy[0]),
        "daughter_1_angle_y": float(daughter_1_angle_xy[1]),
        "daughter_2_angle_x": float(daughter_2_angle_xy[0]),
        "daughter_2_angle_y": float(daughter_2_angle_xy[1]),
        "branching_coefficient": branching_coefficient,
        "vessel_type": vessel_type,
        "parent_path_length_px": parent_length,
        "daughter_1_path_length_px": daughter_1_length,
        "daughter_2_path_length_px": daughter_2_length,
        "n_parent_width_samples": n_parent,
        "n_daughter_1_width_samples": n_daughter_1,
        "n_daughter_2_width_samples": n_daughter_2,
    }


def _branching_records_for_image(
    image_id: str,
    vessel_mask: np.ndarray,
    vessel_type: str,
    disc_center_xy: np.ndarray,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    disc_radius_px: float,
    config: VesselBranchingConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not np.any(vessel_mask):
        return [], []

    inner_radius_px = float(disc_radius_px * inner_circle.diameter)
    outer_radius_px = float(disc_radius_px * outer_circle.diameter)
    branching_points = trace_vessel_branching_points_between_disc_circle_pair(
        vessel_mask=vessel_mask,
        disc_center_xy=disc_center_xy,
        inner_radius_px=inner_radius_px,
        outer_radius_px=outer_radius_px,
        boundary_tolerance_px=config.boundary_tolerance_px,
    )

    branching_records: list[dict[str, object]] = []
    width_records: list[dict[str, object]] = []
    for branching_point in branching_points:
        parent_records = _branch_width_sample_records(
            image_id=image_id,
            vessel_type=vessel_type,
            inner_circle=inner_circle,
            outer_circle=outer_circle,
            inner_radius_px=inner_radius_px,
            outer_radius_px=outer_radius_px,
            branching_point=branching_point,
            branch_role="parent",
            path_xy=branching_point.parent_path_xy,
            vessel_mask=vessel_mask,
            config=config,
        )
        daughter_1_records = _branch_width_sample_records(
            image_id=image_id,
            vessel_type=vessel_type,
            inner_circle=inner_circle,
            outer_circle=outer_circle,
            inner_radius_px=inner_radius_px,
            outer_radius_px=outer_radius_px,
            branching_point=branching_point,
            branch_role="daughter_1",
            path_xy=branching_point.daughter_paths_xy[0],
            vessel_mask=vessel_mask,
            config=config,
        )
        daughter_2_records = _branch_width_sample_records(
            image_id=image_id,
            vessel_type=vessel_type,
            inner_circle=inner_circle,
            outer_circle=outer_circle,
            inner_radius_px=inner_radius_px,
            outer_radius_px=outer_radius_px,
            branching_point=branching_point,
            branch_role="daughter_2",
            path_xy=branching_point.daughter_paths_xy[1],
            vessel_mask=vessel_mask,
            config=config,
        )

        if not parent_records or not daughter_1_records or not daughter_2_records:
            continue

        width_records.extend(parent_records)
        width_records.extend(daughter_1_records)
        width_records.extend(daughter_2_records)
        branching_records.append(
            _branching_record(
                image_id=image_id,
                vessel_type=vessel_type,
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                inner_radius_px=inner_radius_px,
                outer_radius_px=outer_radius_px,
                branching_point=branching_point,
                parent_records=parent_records,
                daughter_1_records=daughter_1_records,
                daughter_2_records=daughter_2_records,
                config=config,
            )
        )

    return branching_records, width_records


def measure_vessel_branching_between_disc_circle_pair(
    vessels_dir: Path,
    av_dir: Path,
    disc_geometry_path: Path,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    output_path: Optional[Path] = None,
    widths_output_path: Optional[Path] = None,
    branching_config: VesselBranchingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure rooted vessel branching geometry between two disc-centered circles."""
    if not disc_geometry_path.exists():
        raise FileNotFoundError(f"Disc geometry file not found: {disc_geometry_path}")
    if not vessels_dir.exists():
        raise FileNotFoundError(f"Vessels directory not found: {vessels_dir}")
    if not av_dir.exists():
        raise FileNotFoundError(f"AV directory not found: {av_dir}")

    config = VesselBranchingConfig() if branching_config is None else branching_config
    df_geometry = pd.read_csv(disc_geometry_path, index_col=0)
    branching_records: list[dict[str, object]] = []
    width_records: list[dict[str, object]] = []
    for image_id, row in df_geometry.iterrows():
        if (
            pd.isna(row["x_disc_center"])
            or pd.isna(row["y_disc_center"])
            or pd.isna(row["disc_radius_px"])
        ):
            continue

        vessel_path = vessels_dir / f"{image_id}.png"
        av_path = av_dir / f"{image_id}.png"
        if not vessel_path.exists() or not av_path.exists():
            continue

        vessel_mask = np.array(Image.open(vessel_path)) > 0
        av_mask = np.array(Image.open(av_path))
        artery_mask, vein_mask = typed_vessel_masks(vessel_mask, av_mask)
        disc_center_xy = np.array(
            [row["x_disc_center"], row["y_disc_center"]],
            dtype=float,
        )

        for typed_mask, vessel_type in ((artery_mask, "artery"), (vein_mask, "vein")):
            image_branching_records, image_width_records = _branching_records_for_image(
                image_id=str(image_id),
                vessel_mask=typed_mask,
                vessel_type=vessel_type,
                disc_center_xy=disc_center_xy,
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                disc_radius_px=float(row["disc_radius_px"]),
                config=config,
            )
            branching_records.extend(image_branching_records)
            width_records.extend(image_width_records)

    df_branching = pd.DataFrame.from_records(
        branching_records,
        columns=VESSEL_BRANCHING_COLUMNS,
    )
    df_widths = pd.DataFrame.from_records(
        width_records,
        columns=VESSEL_BRANCHING_WIDTH_COLUMNS,
    )
    if output_path is not None:
        df_branching.to_csv(output_path, index=False)
    if widths_output_path is not None:
        df_widths.to_csv(widths_output_path, index=False)
    return df_branching, df_widths
