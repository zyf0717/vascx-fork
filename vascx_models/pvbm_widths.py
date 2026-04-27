from __future__ import annotations

import math

import numpy as np


def _nan_point() -> np.ndarray:
    return np.array([float("nan"), float("nan")], dtype=float)


def _normal_from_tangent(tangent_xy: np.ndarray) -> np.ndarray | None:
    normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
    norm = float(np.linalg.norm(normal_xy))
    if norm == 0.0:
        return None
    return normal_xy / norm


def _rounded_coordinate(point_xy: np.ndarray) -> tuple[int, int]:
    return int(np.rint(point_xy[0])), int(np.rint(point_xy[1]))


def _mask_contains(mask: np.ndarray, coordinate_xy: tuple[int, int]) -> bool:
    x, y = coordinate_xy
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])


def _trace_integer_half_width(
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> float:
    center_coordinate = _rounded_coordinate(center_xy)
    if not _mask_contains(vessel_mask, center_coordinate):
        return float("nan")

    last_inside_distance = 0.0
    visited_coordinates = {center_coordinate}
    max_steps = int(math.ceil(float(np.hypot(*vessel_mask.shape)))) + 2

    for step in range(1, max_steps + 1):
        coordinate = _rounded_coordinate(center_xy + direction_xy * float(step))
        if coordinate in visited_coordinates:
            continue
        visited_coordinates.add(coordinate)
        if not _mask_contains(vessel_mask, coordinate):
            return last_inside_distance + 0.5
        last_inside_distance = float(step)

    return float("nan")


def measure_pvbm_mask_width(
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    max_asymmetry_px: float = 1.0,
) -> dict[str, object]:
    normal_xy = _normal_from_tangent(tangent_xy)
    result = {
        "width_method": "pvbm_mask",
        "width_px": float("nan"),
        "x_start": float("nan"),
        "y_start": float("nan"),
        "x_end": float("nan"),
        "y_end": float("nan"),
        "normal_x": float("nan"),
        "normal_y": float("nan"),
        "mask_width_px": float("nan"),
        "measurement_valid": False,
        "measurement_failure_reason": None,
    }
    if normal_xy is None:
        result["measurement_failure_reason"] = "normal_unavailable"
        return result

    result["normal_x"] = float(normal_xy[0])
    result["normal_y"] = float(normal_xy[1])

    positive = _trace_integer_half_width(vessel_mask, center_xy, normal_xy)
    negative = _trace_integer_half_width(vessel_mask, center_xy, -normal_xy)
    if np.isnan(positive) or np.isnan(negative):
        result["measurement_failure_reason"] = "pvbm_mask_boundary_not_found"
        return result

    if abs(positive - negative) > max_asymmetry_px:
        result["measurement_failure_reason"] = "pvbm_mask_asymmetry_exceeds_threshold"
        return result

    start_xy = center_xy - normal_xy * negative
    end_xy = center_xy + normal_xy * positive
    width_px = positive + negative
    result.update(
        {
            "width_px": float(width_px),
            "x_start": float(start_xy[0]),
            "y_start": float(start_xy[1]),
            "x_end": float(end_xy[0]),
            "y_end": float(end_xy[1]),
            "mask_width_px": float(width_px),
            "measurement_valid": True,
            "measurement_failure_reason": None,
        }
    )
    return result
