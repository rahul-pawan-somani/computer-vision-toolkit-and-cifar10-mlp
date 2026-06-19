from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import DATA_DIR, OUTPUTS_DIR, save_text


def _quantize_channel(ch: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Quantize a uint8 channel into bin indices in [0, num_bins - 1].
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    # Map intensities 0..255 to bin indices 0..num_bins-1.
    idx = (ch.astype(np.int32) * num_bins) // 256
    idx = np.clip(idx, 0, num_bins - 1)
    return idx.astype(np.int32)


def _normalised_histogram_from_channel(ch: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Compute a normalised histogram for one uint8 image channel.
    """
    idx = _quantize_channel(ch, num_bins).ravel()
    hist = np.zeros(num_bins, dtype=np.float64)

    for b in idx:
        hist[int(b)] += 1.0

    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def spatial_rgb_histogram(
    image: np.ndarray, num_bins: int = 8, grid: Tuple[int, int] = (2, 2)
) -> np.ndarray:
    """
    Build a spatial RGB histogram by splitting the image into a grid,
    computing an RGB histogram in each cell, and concatenating them.

    Returns:
        Feature vector of shape (gh * gw * 3 * num_bins,)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    gh, gw = grid
    if gh <= 0 or gw <= 0:
        raise ValueError("grid dims must be positive")

    H, W, _ = image.shape
    cell_h = H // gh
    cell_w = W // gw
    if cell_h == 0 or cell_w == 0:
        raise ValueError("grid too fine for image size")

    feats = []

    for r in range(gh):
        for c in range(gw):
            y0, y1 = r * cell_h, (r + 1) * cell_h if r < gh - 1 else H
            x0, x1 = c * cell_w, (c + 1) * cell_w if c < gw - 1 else W
            patch = image[y0:y1, x0:x1, :]

            for ch in range(3):
                hist = _normalised_histogram_from_channel(patch[:, :, ch], num_bins)
                feats.append(hist)

    return np.concatenate(feats, axis=0)


def load_flower() -> np.ndarray:
    """
    Load data/flower.jpg as an RGB NumPy array.
    """
    path = DATA_DIR / "flower.jpg"
    im = Image.open(path).convert("RGB")
    return np.array(im)


def save_spatial_histogram_plot(image: np.ndarray, num_bins: int, grid: Tuple[int, int]) -> None:
    """
    Save a 2x2 per-cell RGB histogram figure for the spatial histogram descriptor.
    Each subplot corresponds to one spatial cell and shows the R, G, B histograms.
    """
    gh, gw = grid
    H, W, _ = image.shape
    cell_h = H // gh
    cell_w = W // gw

    fig, axes = plt.subplots(gh, gw, figsize=(12, 8), sharex=True, sharey=True)
    axes = np.array(axes, dtype=object).reshape(gh, gw)

    bin_positions = np.arange(num_bins)

    for r in range(gh):
        for c in range(gw):
            y0, y1 = r * cell_h, (r + 1) * cell_h if r < gh - 1 else H
            x0, x1 = c * cell_w, (c + 1) * cell_w if c < gw - 1 else W
            patch = image[y0:y1, x0:x1, :]

            ax = axes[r, c]

            channel_names = ["Red", "Green", "Blue"]
            channel_colours = ["red", "green", "blue"]

            for ch in range(3):
                hist = _normalised_histogram_from_channel(patch[:, :, ch], num_bins)

                ax.plot(
                    bin_positions,
                    hist,
                    marker="o",
                    color=channel_colours[ch],
                    label=channel_names[ch],
                    linewidth=2,
                )

            ax.set_title(f"Cell ({r},{c})")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Normalised count")
            ax.set_xticks(bin_positions)
            ax.legend()
            ax.grid(alpha=0.3)

    fig.suptitle(f"Q2 Spatial RGB Histograms per Cell (grid={gh}x{gw}, num_bins={num_bins})")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "q2_spatial_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    img = load_flower()
    num_bins = 8
    grid = (2, 2)

    feat = spatial_rgb_histogram(img, num_bins=num_bins, grid=grid)
    np.save(OUTPUTS_DIR / "q2_spatial_hist.npy", feat)
    save_spatial_histogram_plot(img, num_bins=num_bins, grid=grid)

    summary = (
        "Q2 spatial histogram demo\n"
        f"image shape: {img.shape}\n"
        f"feature shape: {feat.shape}\n"
        f"feature dtype: {feat.dtype}\n"
        f"feature sum: {feat.sum():.6f} "
        "(4 cells × 3 channel histograms, each normalised to sum to 1)\n"
        "saved descriptor: q2_spatial_hist.npy\n"
        "saved figure: q2_spatial_hist.png\n"
        "saved summary: q2_spatial_hist_summary.txt\n"
    )
    print(summary)
    save_text(OUTPUTS_DIR / "q2_spatial_hist_summary.txt", summary)


if __name__ == "__main__":
    main()