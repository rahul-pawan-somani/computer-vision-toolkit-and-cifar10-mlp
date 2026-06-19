from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import OUTPUTS_DIR, save_csv, save_text


def create_diagonal_edge_image(size: int = 9) -> np.ndarray:
    """
    Create a size x size grayscale image with a diagonal edge.

    Pixels on or above the main diagonal are set to 255.
    Pixels below the diagonal are set to 0.

    Returns:
        A uint8 NumPy array of shape (size, size).
    """
    if size <= 0:
        raise ValueError("size must be positive")

    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if j >= i:
                img[i, j] = 255
    return img


def _check_inside_3x3(img: np.ndarray, x: int, y: int) -> None:
    """
    Check that pixel (x, y) has a full 3x3 neighbourhood inside the image.
    """
    if x <= 0 or y <= 0 or x >= img.shape[0] - 1 or y >= img.shape[1] - 1:
        raise ValueError("x, y must have a full 3x3 neighbourhood inside the image")
    

def _finite_diff_3x3(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """
    Estimate the x and y partial derivatives at (x, y)
    using central differences over a 3x3 neighbourhood.

    Gx = (right - left) / 2
    Gy = (down - up) / 2
    """
    _check_inside_3x3(img, x, y)

    left = float(img[x, y - 1])
    right = float(img[x, y + 1])
    up = float(img[x - 1, y])
    down = float(img[x + 1, y])

    gx = (right - left) / 2.0
    gy = (down - up) / 2.0
    return gx, gy


def compute_custom_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """
    Compute gradient magnitude and direction at (x, y)
    using the custom finite-difference method.

    Returns:
        magnitude: Gradient strength.
        direction: Gradient angle in degrees.
    """
    gx, gy = _finite_diff_3x3(img, x, y)
    magnitude = float((gx * gx + gy * gy) ** 0.5)
    direction = float(np.degrees(np.arctan2(gy, gx)))
    return magnitude, direction


def _sobel_like(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """
    Compute Sobel-style horizontal and vertical gradient components
    at pixel (x, y) using the surrounding 3x3 neighbourhood.
    """
    _check_inside_3x3(img, x, y)

    p = img[x - 1 : x + 2, y - 1 : y + 2].astype(np.float32)

    gx = (
        -1 * p[0, 0] + 0 * p[0, 1] + 1 * p[0, 2]
        + -2 * p[1, 0] + 0 * p[1, 1] + 2 * p[1, 2]
        + -1 * p[2, 0] + 0 * p[2, 1] + 1 * p[2, 2]
    )

    gy = (
        -1 * p[0, 0] - 2 * p[0, 1] - 1 * p[0, 2]
        + 0 * p[1, 0] + 0 * p[1, 1] + 0 * p[1, 2]
        + 1 * p[2, 0] + 2 * p[2, 1] + 1 * p[2, 2]
    )

    return gx, gy


def compute_diagonal_corrected_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """
    Estimate gradient magnitude and direction using
    both axis-aligned and diagonal edge responses.

    This is intended to reduce the directional bias of
    standard Sobel filtering on diagonal edges.
    """
    _check_inside_3x3(img, x, y)

    p = img[x - 1 : x + 2, y - 1 : y + 2].astype(np.float32)

    gx_s, gy_s = _sobel_like(img, x, y)
    mag_axis = float((gx_s * gx_s + gy_s * gy_s) ** 0.5)

    # Add diagonal-oriented responses so diagonal edges are not judged
    # only by horizontal and vertical Sobel components.
    g45 = (
        -1 * p[0, 0] - 2 * p[0, 1] + 0 * p[0, 2]
        + -2 * p[1, 0] + 0 * p[1, 1] + 2 * p[1, 2]
        + 0 * p[2, 0] + 2 * p[2, 1] + 1 * p[2, 2]
    )

    g135 = (
        0 * p[0, 0] + 2 * p[0, 1] + 1 * p[0, 2]
        + -2 * p[1, 0] + 0 * p[1, 1] + 2 * p[1, 2]
        + -1 * p[2, 0] - 2 * p[2, 1] + 0 * p[2, 2]
    )

    mag_45 = abs(float(g45))
    mag_135 = abs(float(g135))

    if mag_45 >= mag_axis and mag_45 >= mag_135:
        magnitude = mag_45
        direction = 45.0
    elif mag_135 >= mag_axis and mag_135 >= mag_45:
        magnitude = mag_135
        direction = 135.0
    else:
        magnitude = mag_axis
        direction = float(np.degrees(np.arctan2(gy_s, gx_s)))

    return magnitude, direction


def save_q1_image(img: np.ndarray) -> None:
    """
    Save the generated Q1 image as both a NumPy array and a plotted PNG.
    """
    np.save(OUTPUTS_DIR / "q1_diagonal_edge.npy", img)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title("Q1 Generated 9x9 Diagonal Edge Image")

    plt.xticks(range(img.shape[1]))
    plt.yticks(range(img.shape[0]))
    plt.grid(color="red", linestyle="-", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "q1_diagonal_edge.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    img = create_diagonal_edge_image(9)
    save_q1_image(img)

    x, y = 4, 4

    mag_c, dir_c = compute_custom_gradient(img, x, y)

    gx_s, gy_s = _sobel_like(img, x, y)
    mag_s = float((gx_s * gx_s + gy_s * gy_s) ** 0.5)
    dir_s = float(np.degrees(np.arctan2(gy_s, gx_s)))

    mag_d, dir_d = compute_diagonal_corrected_gradient(img, x, y)

    out = []
    out.append("Q1 numeric comparison at (4,4)\n")
    out.append(f"Custom finite-diff: magnitude={mag_c:.4f}, direction={dir_c:.2f} deg")
    out.append(f"Sobel-like:        magnitude={mag_s:.4f}, direction={dir_s:.2f} deg")
    out.append(f"Diag-corrected:    magnitude={mag_d:.4f}, direction={dir_d:.2f} deg")
    out.append("")
    out.append("Saved outputs:")
    out.append("q1_diagonal_edge.npy")
    out.append("q1_diagonal_edge.png")
    out.append("q1_gradients.csv")
    out.append("q1_gradients.txt")
    text = "\n".join(out) + "\n"

    print(text)
    save_text(OUTPUTS_DIR / "q1_gradients.txt", text)

    csv_rows = [
        f"Custom finite difference,{mag_c:.4f},{dir_c:.2f}",
        f"Sobel-like,{mag_s:.4f},{dir_s:.2f}",
        f"Diagonal-corrected,{mag_d:.4f},{dir_d:.2f}",
    ]
    save_csv(
        OUTPUTS_DIR / "q1_gradients.csv",
        header="method,magnitude,direction_degrees",
        rows=csv_rows,
    )


if __name__ == "__main__":
    main()