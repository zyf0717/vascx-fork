from __future__ import annotations

import numpy as np

from .config import OverlayCircle

VESSEL_TORTUOSITY_COLUMNS = [
    "image_id",
    "inner_circle",
    "outer_circle",
    "inner_circle_radius_px",
    "outer_circle_radius_px",
    "connection_index",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "path_length_px",
    "chord_length_px",
    "tortuosity",
    "vessel_type",
]


def compute_path_tortuosity(path_xy: np.ndarray) -> tuple[float, float, float]:
    """Return path length, chord length, and tortuosity for an ordered path."""
    if len(path_xy) < 2:
        return 0.0, 0.0, float("nan")

    deltas = np.diff(path_xy, axis=0)
    path_length_px = float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())
    chord_length_px = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
    tortuosity = (
        path_length_px / chord_length_px
        if chord_length_px > 0.0
        else float("nan")
    )
    return path_length_px, chord_length_px, float(tortuosity)


def vessel_tortuosity_record(
    image_id: str,
    vessel_type: str,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    inner_radius_px: float,
    outer_radius_px: float,
    connection_index: int,
    path_xy: np.ndarray,
) -> dict[str, object]:
    """Build a one-row vessel tortuosity record from an ordered skeleton path."""
    path_length_px, chord_length_px, tortuosity = compute_path_tortuosity(path_xy)
    return {
        "image_id": image_id,
        "inner_circle": inner_circle.name,
        "outer_circle": outer_circle.name,
        "inner_circle_radius_px": inner_radius_px,
        "outer_circle_radius_px": outer_radius_px,
        "connection_index": connection_index,
        "x_start": float(path_xy[0, 0]),
        "y_start": float(path_xy[0, 1]),
        "x_end": float(path_xy[-1, 0]),
        "y_end": float(path_xy[-1, 1]),
        "path_length_px": path_length_px,
        "chord_length_px": chord_length_px,
        "tortuosity": tortuosity,
        "vessel_type": vessel_type,
    }
