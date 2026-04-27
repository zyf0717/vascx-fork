import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .config import OverlayCircle
from .vessel_paths import (
    interpolate_path_point,
    path_cumulative_lengths,
    skeletonize,
    trace_vessel_paths_between_disc_circle_pair,
)
from .vessel_tortuosities import (
    VESSEL_TORTUOSITY_COLUMNS,
    vessel_tortuosity_record,
)

logger = logging.getLogger(__name__)

VESSEL_WIDTH_COLUMNS = [
    "image_id",
    "inner_circle",
    "outer_circle",
    "inner_circle_radius_px",
    "outer_circle_radius_px",
    "connection_index",
    "sample_index",
    "x",
    "y",
    "width_px",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "vessel_type",
]


def _local_skeleton_points(
    skeleton: np.ndarray,
    point_xy: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    x, y = point_xy
    height, width = skeleton.shape
    x_min = max(0, int(np.floor(x - radius_px)))
    x_max = min(width - 1, int(np.ceil(x + radius_px)))
    y_min = max(0, int(np.floor(y - radius_px)))
    y_max = min(height - 1, int(np.ceil(y + radius_px)))

    window = skeleton[y_min : y_max + 1, x_min : x_max + 1]
    ys, xs = np.nonzero(window)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)

    xs = xs.astype(float) + float(x_min)
    ys = ys.astype(float) + float(y_min)
    deltas = np.column_stack((xs - x, ys - y))
    keep = np.sum(deltas * deltas, axis=1) <= radius_px * radius_px
    return np.column_stack((xs[keep], ys[keep]))


def _estimate_tangent(points_xy: np.ndarray) -> Optional[np.ndarray]:
    if len(points_xy) < 2:
        return None

    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        return None
    return tangent / norm


def _sample_mask(mask: np.ndarray, point_xy: np.ndarray) -> float:
    x, y = point_xy
    height, width = mask.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    v00 = float(mask[y0, x0])
    v10 = float(mask[y0, x1])
    v01 = float(mask[y1, x0])
    v11 = float(mask[y1, x1])

    top = v00 * (1.0 - dx) + v10 * dx
    bottom = v01 * (1.0 - dx) + v11 * dx
    return top * (1.0 - dy) + bottom * dy


def _trace_boundary_distance(
    vessel_mask: np.ndarray,
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    step_px: float,
) -> float:
    max_distance = float(np.hypot(*vessel_mask.shape)) + 2.0
    current_value = _sample_mask(vessel_mask, point_xy)
    if current_value < 0.5:
        return float("nan")

    t_inside = 0.0
    steps = int(np.ceil(max_distance / step_px))
    for step in range(1, steps + 1):
        t_outside = step * step_px
        sample_point = point_xy + direction_xy * t_outside
        if _sample_mask(vessel_mask, sample_point) < 0.5:
            low = t_inside
            high = t_outside
            for _ in range(12):
                mid = (low + high) / 2.0
                mid_point = point_xy + direction_xy * mid
                if _sample_mask(vessel_mask, mid_point) >= 0.5:
                    low = mid
                else:
                    high = mid
            return low
        t_inside = t_outside

    return float("nan")


def _boundary_point(
    center_xy: np.ndarray,
    direction_xy: np.ndarray,
    distance_px: float,
) -> np.ndarray:
    return center_xy + direction_xy * distance_px


def _measure_width_along_normal(
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    step_px: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
    norm = float(np.linalg.norm(normal_xy))
    if norm == 0.0:
        nan_point = np.array([float("nan"), float("nan")], dtype=float)
        return float("nan"), nan_point, nan_point
    normal_xy /= norm

    positive = _trace_boundary_distance(vessel_mask, center_xy, normal_xy, step_px)
    negative = _trace_boundary_distance(vessel_mask, center_xy, -normal_xy, step_px)
    if np.isnan(positive) or np.isnan(negative):
        nan_point = np.array([float("nan"), float("nan")], dtype=float)
        return float("nan"), nan_point, nan_point

    start_xy = _boundary_point(center_xy, -normal_xy, negative)
    end_xy = _boundary_point(center_xy, normal_xy, positive)
    return positive + negative, start_xy, end_xy


def measure_vessel_width_at_coordinate(
    vessel_mask: np.ndarray,
    point_xy: np.ndarray,
    skeleton: Optional[np.ndarray] = None,
    tangent_window_px: float = 10.0,
    measurement_step_px: float = 0.25,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Measure vessel width at an arbitrary coordinate using the local skeleton tangent."""
    skeleton = skeletonize(vessel_mask) if skeleton is None else skeleton
    local_points = _local_skeleton_points(skeleton, point_xy, tangent_window_px)
    tangent_xy = _estimate_tangent(local_points)
    if tangent_xy is None:
        nan_point = np.array([float("nan"), float("nan")], dtype=float)
        return float("nan"), nan_point, nan_point

    return _measure_width_along_normal(
        vessel_mask=vessel_mask,
        center_xy=point_xy,
        tangent_xy=tangent_xy,
        step_px=measurement_step_px,
    )


def _typed_vessel_masks(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the vessel mask into artery and vein masks using the AV segmentation."""
    artery_mask = vessel_mask & np.isin(av_mask, (1, 3))
    vein_mask = vessel_mask & np.isin(av_mask, (2, 3))
    return artery_mask, vein_mask


def _path_records_for_image(
    image_id: str,
    vessel_mask: np.ndarray,
    vessel_type: str,
    disc_center_xy: np.ndarray,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    disc_radius_px: float,
    samples_per_connection: int,
    boundary_tolerance_px: float,
    tangent_window_px: float,
    measurement_step_px: float,
) -> tuple[List[dict], List[dict]]:
    if not np.any(vessel_mask):
        return [], []

    if samples_per_connection <= 0:
        raise ValueError("samples_per_connection must be positive")

    inner_radius_px = float(disc_radius_px * inner_circle.diameter)
    outer_radius_px = float(disc_radius_px * outer_circle.diameter)
    if outer_radius_px <= inner_radius_px:
        raise ValueError("outer_circle must have a larger radius than inner_circle")

    vessel_paths = trace_vessel_paths_between_disc_circle_pair(
        vessel_mask=vessel_mask,
        disc_center_xy=disc_center_xy,
        inner_radius_px=inner_radius_px,
        outer_radius_px=outer_radius_px,
        boundary_tolerance_px=boundary_tolerance_px,
    )

    width_records: List[dict] = []
    tortuosity_records: List[dict] = []
    for vessel_path in vessel_paths:
        cumulative_lengths = path_cumulative_lengths(vessel_path.path_xy)
        component_records: List[dict] = []
        for sample_index in range(1, samples_per_connection + 1):
            target_fraction = sample_index / (samples_per_connection + 1)
            center_xy = interpolate_path_point(
                path_xy=vessel_path.path_xy,
                cumulative_lengths=cumulative_lengths,
                target_length=float(cumulative_lengths[-1] * target_fraction),
            )
            width_value, start_xy, end_xy = measure_vessel_width_at_coordinate(
                vessel_mask=vessel_mask,
                point_xy=center_xy,
                skeleton=vessel_path.skeleton,
                tangent_window_px=tangent_window_px,
                measurement_step_px=measurement_step_px,
            )
            if np.isnan(width_value):
                # TODO: Recover from local tangent/width failures by re-sampling nearby
                # skeleton points.
                component_records = []
                logger.debug(
                    "Skipping %s connection %d because width measurement failed at sample %d",
                    image_id,
                    vessel_path.connection_index,
                    sample_index,
                )
                break

            component_records.append(
                {
                    "image_id": image_id,
                    "inner_circle": inner_circle.name,
                    "outer_circle": outer_circle.name,
                    "inner_circle_radius_px": inner_radius_px,
                    "outer_circle_radius_px": outer_radius_px,
                    "connection_index": vessel_path.connection_index,
                    "sample_index": sample_index,
                    "x": float(center_xy[0]),
                    "y": float(center_xy[1]),
                    "width_px": float(width_value),
                    "x_start": float(start_xy[0]),
                    "y_start": float(start_xy[1]),
                    "x_end": float(end_xy[0]),
                    "y_end": float(end_xy[1]),
                    "vessel_type": vessel_type,
                }
            )

        if len(component_records) != samples_per_connection:
            continue
        width_records.extend(component_records)
        tortuosity_records.append(
            vessel_tortuosity_record(
                image_id=image_id,
                vessel_type=vessel_type,
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                inner_radius_px=inner_radius_px,
                outer_radius_px=outer_radius_px,
                connection_index=vessel_path.connection_index,
                path_xy=vessel_path.path_xy,
            )
        )

    return width_records, tortuosity_records


def resolve_vessel_width_circle_pair(
    circles: Sequence[OverlayCircle],
    inner_circle_name: str | None = None,
    outer_circle_name: str | None = None,
) -> tuple[OverlayCircle, OverlayCircle]:
    """Select the circle pair used for between-circle vessel width sampling."""
    if len(circles) < 2:
        raise ValueError(
            "At least two overlay circles are required for vessel width sampling"
        )

    circles_by_name = {circle.name: circle for circle in circles}
    if inner_circle_name is not None:
        if inner_circle_name not in circles_by_name:
            raise ValueError(f"Unknown inner circle '{inner_circle_name}'")
        inner_circle = circles_by_name[inner_circle_name]
    else:
        inner_circle = min(circles, key=lambda circle: (circle.diameter, circle.name))

    if outer_circle_name is not None:
        if outer_circle_name not in circles_by_name:
            raise ValueError(f"Unknown outer circle '{outer_circle_name}'")
        outer_circle = circles_by_name[outer_circle_name]
    else:
        remaining_circles = [
            circle for circle in circles if circle.name != inner_circle.name
        ]
        larger_circles = [
            circle
            for circle in remaining_circles
            if circle.diameter > inner_circle.diameter
        ]
        candidates = larger_circles if larger_circles else remaining_circles
        outer_circle = min(
            candidates, key=lambda circle: (circle.diameter, circle.name)
        )

    if outer_circle.diameter <= inner_circle.diameter:
        raise ValueError("outer_circle must have a larger diameter than inner_circle")
    return inner_circle, outer_circle


def measure_vessel_widths_and_tortuosities_between_disc_circle_pair(
    vessels_dir: Path,
    av_dir: Path,
    disc_geometry_path: Path,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    output_path: Optional[Path] = None,
    tortuosity_output_path: Optional[Path] = None,
    samples_per_connection: int = 5,
    boundary_tolerance_px: float = 1.5,
    tangent_window_px: float = 10.0,
    measurement_step_px: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure vessel widths and path tortuosities between two circles."""
    if not disc_geometry_path.exists():
        raise FileNotFoundError(f"Disc geometry file not found: {disc_geometry_path}")
    if not vessels_dir.exists():
        raise FileNotFoundError(f"Vessels directory not found: {vessels_dir}")
    if not av_dir.exists():
        raise FileNotFoundError(f"AV directory not found: {av_dir}")

    df_geometry = pd.read_csv(disc_geometry_path, index_col=0)
    width_records: List[dict] = []
    tortuosity_records: List[dict] = []

    for image_id, row in df_geometry.iterrows():
        if (
            pd.isna(row["x_disc_center"])
            or pd.isna(row["y_disc_center"])
            or pd.isna(row["disc_radius_px"])
        ):
            logger.warning("Skipping %s because disc geometry is missing", image_id)
            continue

        vessel_path = vessels_dir / f"{image_id}.png"
        av_path = av_dir / f"{image_id}.png"
        if not vessel_path.exists() or not av_path.exists():
            logger.warning("Skipping %s because vessel or AV mask is missing", image_id)
            continue

        vessel_mask = np.array(Image.open(vessel_path)) > 0
        av_mask = np.array(Image.open(av_path))
        artery_mask, vein_mask = _typed_vessel_masks(vessel_mask, av_mask)
        for typed_mask, vessel_type in ((artery_mask, "artery"), (vein_mask, "vein")):
            image_width_records, image_tortuosity_records = _path_records_for_image(
                image_id=image_id,
                vessel_mask=typed_mask,
                vessel_type=vessel_type,
                disc_center_xy=np.array(
                    [row["x_disc_center"], row["y_disc_center"]], dtype=float
                ),
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                disc_radius_px=float(row["disc_radius_px"]),
                samples_per_connection=samples_per_connection,
                boundary_tolerance_px=boundary_tolerance_px,
                tangent_window_px=tangent_window_px,
                measurement_step_px=measurement_step_px,
            )
            width_records.extend(image_width_records)
            tortuosity_records.extend(image_tortuosity_records)

    df_widths = pd.DataFrame.from_records(
        width_records,
        columns=VESSEL_WIDTH_COLUMNS,
    )
    df_tortuosities = pd.DataFrame.from_records(
        tortuosity_records,
        columns=VESSEL_TORTUOSITY_COLUMNS,
    )
    if output_path is not None:
        df_widths.to_csv(output_path, index=False)
        logger.info("Vessel path width measurements saved to %s", output_path)
    if tortuosity_output_path is not None:
        df_tortuosities.to_csv(tortuosity_output_path, index=False)
        logger.info("Vessel path tortuosities saved to %s", tortuosity_output_path)
    return df_widths, df_tortuosities


def measure_vessel_widths_between_disc_circle_pair(
    vessels_dir: Path,
    av_dir: Path,
    disc_geometry_path: Path,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    output_path: Optional[Path] = None,
    samples_per_connection: int = 5,
    boundary_tolerance_px: float = 1.5,
    tangent_window_px: float = 10.0,
    measurement_step_px: float = 0.25,
) -> pd.DataFrame:
    """Measure artery and vein widths separately at interior points between two circles."""
    df_widths, _ = measure_vessel_widths_and_tortuosities_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=disc_geometry_path,
        inner_circle=inner_circle,
        outer_circle=outer_circle,
        output_path=output_path,
        samples_per_connection=samples_per_connection,
        boundary_tolerance_px=boundary_tolerance_px,
        tangent_window_px=tangent_window_px,
        measurement_step_px=measurement_step_px,
    )
    return df_widths


def revised_vessel_equivalent(
    diameters: Iterable[float],
    vessel_type: str,
    n_largest: int = 6,
    return_rounds: bool = False,
):
    """
    Compute revised CRAE-6 or CRVE-6 using the Knudtson revised formula.

    Parameters
    ----------
    diameters:
        Vessel diameters in consistent units: pixels, microns, etc.
    vessel_type:
        Accepts 'artery'|'arteriole' or 'vein'|'venule'.
    n_largest:
        Number of largest vessels to use. Standard revised method uses 6.
    return_rounds:
        If True, also return intermediate rounds for inspection.

    Returns
    -------
    float
        Final equivalent vessel diameter.
    """

    coeffs = {
        "arteriole": 0.88,
        "venule": 0.95,
    }

    # Accept synonyms
    if vessel_type in ("artery", "arteriole"):
        vt = "arteriole"
    elif vessel_type in ("vein", "venule"):
        vt = "venule"
    else:
        raise ValueError("vessel_type must be 'arteriole'/'artery' or 'venule'/'vein'.")

    coeff = coeffs[vt]

    values = sorted([float(x) for x in diameters if float(x) > 0.0], reverse=True)[
        :n_largest
    ]

    if len(values) < 2:
        raise ValueError("Need at least two valid vessel diameters.")

    rounds: list[list[float]] = [values.copy()]

    while len(values) > 1:
        values = sorted(values, reverse=True)

        next_values: list[float] = []

        # Pair largest with smallest.
        while len(values) >= 2:
            largest = values.pop(0)
            smallest = values.pop(-1)

            combined = coeff * math.sqrt(largest**2 + smallest**2)
            next_values.append(combined)

        # If odd count remains, carry the middle value forward unchanged.
        if values:
            next_values.append(values[0])

        values = next_values
        rounds.append(values.copy())

    result = float(values[0])

    if return_rounds:
        return result, rounds

    return result


def compute_revised_crx_from_widths(
    df_widths: pd.DataFrame,
    n_largest: int = 6,
    return_rounds: bool = False,
    df_tortuosities: pd.DataFrame | None = None,
):
    """Aggregate per-connection mean widths and compute revised CRAE/CRVE.

    Parameters
    ----------
    df_widths:
        DataFrame produced by `measure_vessel_widths_between_disc_circle_pair` containing
        per-sample `width_px` values and `connection_index` identifiers.
    n_largest:
        Number of largest vessel averages to consider (default 6).
    return_rounds:
        If True, also return the intermediate rounds from the revised algorithm
        for each image/vessel_type as a mapping.
    df_tortuosities:
        Optional one-row-per-vessel tortuosity table produced by
        `measure_vessel_widths_and_tortuosities_between_disc_circle_pair`. When
        provided, the equivalent output includes the mean tortuosity of the
        selected top-width vessels.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] or tuple[pd.DataFrame, pd.DataFrame, dict]
        Returns `(df_connections, df_equivalents)` or
        `(df_connections, df_equivalents, rounds_map)` when `return_rounds` is True.
        `df_connections` contains the per-connection mean widths and selection flags.
    """

    include_tortuosity = df_tortuosities is not None

    cols_conn = [
        "image_id",
        "vessel_type",
        "connection_index",
        "vessel_id",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
        "mean_width_px",
        "n_samples",
        "selected_for_equivalent",
    ]
    if include_tortuosity:
        cols_conn.append("tortuosity")
    cols_equiv = [
        "image_id",
        "metric",
        "vessel_type",
        "requested_n_largest",
        "n_vessels_available",
        "n_vessels_used",
        "vessel_ids_used",
        "mean_widths_used_px",
        "equivalent_px",
    ]
    if include_tortuosity:
        cols_equiv.append("mean_tortuosity_used")

    if df_widths.empty:
        df_conn_empty = pd.DataFrame(columns=cols_conn)
        df_equiv_empty = pd.DataFrame(columns=cols_equiv)
        if return_rounds:
            return df_conn_empty, df_equiv_empty, {}
        return df_conn_empty, df_equiv_empty

    group_cols = [
        "image_id",
        "vessel_type",
        "connection_index",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
    ]

    df_conn = (
        df_widths.groupby(group_cols, dropna=False)
        .agg(mean_width_px=("width_px", "mean"), n_samples=("width_px", "count"))
        .reset_index()
    )
    df_conn["vessel_id"] = df_conn.apply(
        lambda row: f"{row['vessel_type']}_{int(row['connection_index'])}",
        axis=1,
    )
    if include_tortuosity:
        tortuosity_lookup_cols = group_cols + ["tortuosity"]
        if df_tortuosities.empty:
            df_conn["tortuosity"] = float("nan")
        else:
            df_conn = df_conn.merge(
                df_tortuosities[tortuosity_lookup_cols],
                on=group_cols,
                how="left",
            )
    df_conn["selected_for_equivalent"] = False
    df_conn = df_conn[cols_conn]

    results: list[dict] = []
    rounds_map: dict[tuple[str, str], list[list[float]]] = {}

    for (image_id, vessel_type), group in df_conn.groupby(["image_id", "vessel_type"]):
        top = group.sort_values(
            ["mean_width_px", "connection_index"],
            ascending=[False, True],
        ).head(n_largest)
        df_conn.loc[top.index, "selected_for_equivalent"] = True
        diameters = top["mean_width_px"].tolist()
        metric = "CRAE" if vessel_type == "artery" else "CRVE"
        result = {
            "image_id": image_id,
            "metric": metric,
            "vessel_type": vessel_type,
            "requested_n_largest": int(n_largest),
            "n_vessels_available": int(len(group)),
            "n_vessels_used": int(len(top)),
            "vessel_ids_used": ";".join(str(value) for value in top["vessel_id"]),
            "mean_widths_used_px": ";".join(
                f"{float(value):.6g}" for value in diameters
            ),
            "equivalent_px": float("nan"),
        }
        if include_tortuosity:
            result["mean_tortuosity_used"] = float(top["tortuosity"].mean())
        try:
            if return_rounds:
                eq, rounds = revised_vessel_equivalent(
                    diameters,
                    vessel_type,
                    n_largest=n_largest,
                    return_rounds=True,
                )
                rounds_map[(image_id, vessel_type)] = rounds
            else:
                eq = revised_vessel_equivalent(
                    diameters, vessel_type, n_largest=n_largest
                )
            result["equivalent_px"] = float(eq)
        except ValueError:
            pass
        results.append(result)

    df_equiv = pd.DataFrame.from_records(results, columns=cols_equiv)

    if return_rounds:
        return df_conn, df_equiv, rounds_map

    return df_conn, df_equiv


def select_vessel_width_measurements_for_equivalents(
    df_widths: pd.DataFrame,
    df_connections: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-sample width rows whose vessels were selected for CRAE/CRVE."""
    if df_widths.empty or df_connections.empty:
        return df_widths.iloc[0:0].copy()

    selected = df_connections[df_connections["selected_for_equivalent"]]
    if selected.empty:
        return df_widths.iloc[0:0].copy()

    merge_cols = [
        "image_id",
        "vessel_type",
        "connection_index",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
    ]
    return df_widths.merge(selected[merge_cols], on=merge_cols, how="inner")
