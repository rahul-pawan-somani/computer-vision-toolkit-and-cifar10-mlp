from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import OUTPUTS_DIR, save_text, save_csv


def compute_inverse_transform(points: np.ndarray, theta: float, t: Tuple[float, float]) -> np.ndarray:
    """
    Compute the inverse rigid transform that maps transformed 2D points
    back to the original points.

    Forward:
        p' = R(theta) (p - C) + C + t

    If the input points are transformed points p', then their centroid is:
        C' = C + t

    Hence:
        C = C' - t

    Inverse:
        p = R(-theta) (p' - C') + C
          = R(-theta) (p' - C') + (C' - t)
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (N,2)")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one point")

    tx, ty = float(t[0]), float(t[1])
    t_vec = np.array([tx, ty], dtype=np.float64)

    # Centroid of transformed points
    C_prime = points.mean(axis=0)

    c = float(np.cos(theta))
    s = float(np.sin(theta))

    # R(-theta)
    R_inv = np.array([[c, s],
                      [-s, c]], dtype=np.float64)

    recovered = ((points - C_prime) @ R_inv.T) + (C_prime - t_vec)
    return recovered


def save_inverse_transform_plot(
    original: np.ndarray,
    transformed: np.ndarray,
    recovered: np.ndarray,
) -> None:
    """
    Save a simple geometric plot showing original, transformed,
    and recovered point sets.
    """
    plt.figure(figsize=(6, 6))

    # Close the shapes for plotting
    original_closed = np.vstack([original, original[0]])
    transformed_closed = np.vstack([transformed, transformed[0]])
    recovered_closed = np.vstack([recovered, recovered[0]])

    plt.plot(
        original_closed[:, 0],
        original_closed[:, 1],
        marker="o",
        linewidth=3,
        color="blue",
        label="Original",
    )

    plt.plot(
        transformed_closed[:, 0],
        transformed_closed[:, 1],
        marker="o",
        linewidth=2,
        color="orange",
        label="Transformed",
    )

    plt.plot(
        recovered_closed[:, 0],
        recovered_closed[:, 1],
        marker="o",
        linestyle="--",
        linewidth=2,
        color="green",
        alpha=0.7,
        label="Recovered",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q4 Inverse Transform Verification")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "q4_inverse_transform_plot.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_points_csv(original: np.ndarray, transformed: np.ndarray, recovered: np.ndarray) -> None:
    """
    Save original, transformed, and recovered point coordinates.
    """
    rows = []

    for p in original:
        rows.append(f"original,{p[0]:.6f},{p[1]:.6f}")

    for p in transformed:
        rows.append(f"transformed,{p[0]:.6f},{p[1]:.6f}")

    for p in recovered:
        rows.append(f"recovered,{p[0]:.6f},{p[1]:.6f}")

    save_csv(
        OUTPUTS_DIR / "q4_points.csv",
        header="set,x,y",
        rows=rows,
    )


def main() -> None:
    pts = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]], dtype=np.float64)
    theta = np.deg2rad(30.0)
    t = (2.0, -1.0)

    C = pts.mean(axis=0)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)
    pts_fwd = ((pts - C) @ R.T) + C + np.array(t, dtype=np.float64)

    pts_rec = compute_inverse_transform(pts_fwd, theta, t)
    err = np.max(np.abs(pts_rec - pts))
    save_points_csv(pts, pts_fwd, pts_rec)
    save_inverse_transform_plot(pts, pts_fwd, pts_rec)

    summary = (
        "Q4 inverse transform sanity check\n"
        f"theta_degrees: {np.rad2deg(theta):.2f}\n"
        f"translation: {t}\n"
        f"max abs reconstruction error: {err:.6e}\n"
        "saved figure: q4_inverse_transform_plot.png\n"
        "saved coordinates: q4_points.csv\n"
        "saved summary: q4_inverse_transform_summary.txt\n"
        "recovered points overlap the original points up to floating-point precision\n"
    )
    print(summary)
    save_text(OUTPUTS_DIR / "q4_inverse_transform_summary.txt", summary)


if __name__ == "__main__":
    main()