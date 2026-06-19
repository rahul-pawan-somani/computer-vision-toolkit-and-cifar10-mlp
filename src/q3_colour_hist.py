from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import DATA_DIR, OUTPUTS_DIR, Timer, save_csv, save_text


def _quantize_channel(ch: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Quantize a uint8 image channel into bin indices in [0, num_bins - 1].
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    idx = (ch.astype(np.int32) * num_bins) // 256
    return np.clip(idx, 0, num_bins - 1).astype(np.int32)


def compute_colour_histogram_numpy(im: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised NumPy implementation without using np.histogram or np.bincount.
    Returns (histR, histG, histB) as NumPy arrays of length num_bins.
    """
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("im must be HxWx3")

    r = _quantize_channel(im[:, :, 0], num_bins).ravel()
    g = _quantize_channel(im[:, :, 1], num_bins).ravel()
    b = _quantize_channel(im[:, :, 2], num_bins).ravel()

    bins = np.arange(num_bins, dtype=np.int32)

    hist_r = (r[:, None] == bins[None, :]).sum(axis=0).astype(np.int64)
    hist_g = (g[:, None] == bins[None, :]).sum(axis=0).astype(np.int64)
    hist_b = (b[:, None] == bins[None, :]).sum(axis=0).astype(np.int64)

    return hist_r, hist_g, hist_b


def compute_colour_histogram_python(im: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure Python loop-based implementation.
    Returns (histR, histG, histB) as NumPy arrays of length num_bins.
    """
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("im must be HxWx3")

    hist_r = [0] * num_bins
    hist_g = [0] * num_bins
    hist_b = [0] * num_bins

    H, W, _ = im.shape
    for y in range(H):
        for x in range(W):
            rv = int(im[y, x, 0])
            gv = int(im[y, x, 1])
            bv = int(im[y, x, 2])

            rb = (rv * num_bins) // 256
            gb = (gv * num_bins) // 256
            bb = (bv * num_bins) // 256

            if rb >= num_bins:
                rb = num_bins - 1
            if gb >= num_bins:
                gb = num_bins - 1
            if bb >= num_bins:
                bb = num_bins - 1

            hist_r[rb] += 1
            hist_g[gb] += 1
            hist_b[bb] += 1

    return (
        np.array(hist_r, dtype=np.int64),
        np.array(hist_g, dtype=np.int64),
        np.array(hist_b, dtype=np.int64),
    )


def load_flower() -> np.ndarray:
    """
    Load data/flower.jpg as an RGB NumPy array.
    """
    path = DATA_DIR / "flower.jpg"
    im = Image.open(path).convert("RGB")
    return np.array(im)


def time_function(func, im: np.ndarray, num_bins: int, repeats: int = 5) -> float:
    """
    Measure the average runtime of a histogram function over several repeats.
    """
    times = []
    for _ in range(repeats):
        with Timer() as t:
            func(im, num_bins)
        times.append(t.elapsed)
    return float(sum(times) / len(times))


def save_runtime_plot(num_bins: list[int], numpy_times: list[float], python_times: list[float]) -> None:
    """
    Save a grouped bar chart comparing NumPy and pure Python runtimes.
    """
    x = np.arange(len(num_bins))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, numpy_times, width=width, label="NumPy", color="blue")
    plt.bar(x + width / 2, python_times, width=width, label="Python", color="red")

    plt.xlabel("Number of bins")
    plt.ylabel("Runtime (seconds)")
    plt.title("Q3 Runtime Comparison: NumPy vs Python")
    plt.xticks(x, [str(b) for b in num_bins])
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for i, v in enumerate(numpy_times):
        plt.text(i - width/2, v + 0.002, f"{v:.3f}", ha="center")

    for i, v in enumerate(python_times):
        plt.text(i + width/2, v + 0.002, f"{v:.3f}", ha="center")
    plt.savefig(OUTPUTS_DIR / "q3_runtimes.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    im = load_flower()
    bins_list = [8, 16, 32]

    rows = []
    numpy_times = []
    python_times = []

    for nb in bins_list:
        hr_n, hg_n, hb_n = compute_colour_histogram_numpy(im, nb)
        hr_p, hg_p, hb_p = compute_colour_histogram_python(im, nb)

        if not (np.array_equal(hr_n, hr_p) and np.array_equal(hg_n, hg_p) and np.array_equal(hb_n, hb_p)):
            raise ValueError(f"Histogram mismatch for num_bins={nb}")

        numpy_t = time_function(compute_colour_histogram_numpy, im, nb, repeats=5)
        py_t = time_function(compute_colour_histogram_python, im, nb, repeats=5)

        numpy_times.append(numpy_t)
        python_times.append(py_t)

        row = f"{nb},{numpy_t:.6f},{py_t:.6f}"
        rows.append(row)

    save_csv(
        OUTPUTS_DIR / "q3_runtimes.csv",
        header="num_bins,numpy_seconds,python_seconds",
        rows=rows,
    )

    save_runtime_plot(bins_list, numpy_times, python_times)

    summary = (
        "Q3 runtime comparison\n"
        "Compared NumPy and pure Python colour histogram implementations.\n"
        f"num_bins tested: {bins_list}\n"
        f"NumPy times: {numpy_times}\n"
        f"Python times: {python_times}\n"
        "Saved outputs: q3_runtimes.csv, q3_runtimes.png, q3_summary.txt\n"
    )
    print(summary)
    save_text(OUTPUTS_DIR / "q3_summary.txt", summary)


if __name__ == "__main__":
    main()