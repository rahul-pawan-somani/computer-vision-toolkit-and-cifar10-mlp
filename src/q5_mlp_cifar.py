from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from utils import DATA_DIR, OUTPUTS_DIR, ensure_dirs, seed_everything


def images_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Divide each image into non-overlapping patches.

    Args:
        x: Tensor of shape (B, 3, 32, 32)
        patch_size: Patch side length, must divide 32

    Returns:
        Tensor of shape (B, num_patches, patch_dim)
    """
    if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] != 32 or x.shape[3] != 32:
        raise ValueError("x must be (B, 3, 32, 32)")
    if 32 % patch_size != 0:
        raise ValueError("patch_size must divide 32")

    b, c, h, w = x.shape
    p = patch_size
    gh = h // p
    gw = w // p

    patches = x.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.reshape(b, gh * gw, c * p * p)


class PatchMLP(nn.Module):
    """
    MLP classifier that projects each image patch with a shared linear layer,
    then flattens all patch embeddings and classifies them with an MLP.
    """

    def __init__(
        self,
        patch_size: int,
        patch_embed_dim: int = 96,
        hidden_dims: Tuple[int, ...] = (1024, 512),
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        num_patches = (32 // patch_size) ** 2

        self.patch_proj = nn.Linear(patch_dim, patch_embed_dim)
        self.patch_norm = nn.LayerNorm(patch_embed_dim)
        self.patch_dropout = nn.Dropout(dropout * 0.5)

        dims = [num_patches * patch_embed_dim] + list(hidden_dims)
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = images_to_patches(x, self.patch_size)   # (B, N, D)
        z = self.patch_proj(patches)                      # (B, N, E)
        z = self.patch_norm(z)
        z = F.gelu(z)
        z = self.patch_dropout(z)
        z = z.reshape(z.shape[0], -1)                     # flatten patch embeddings
        return self.classifier(z)


@dataclass
class TrainConfig:
    patch_size: int = 4
    patch_embed_dim: int = 96
    hidden_dims: Tuple[int, ...] = (1024, 512)
    dropout: float = 0.2
    batch_size: int = 256
    lr: float = 3e-4
    epochs: int = 40
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    val_split: float = 0.1
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    label_smoothing: float,
) -> float:
    model.train()
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()

    return total_correct / total


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()

    return total_correct / total


@torch.no_grad()
def test_mlp(model: nn.Module, test_loader: DataLoader, device: str) -> Tuple[np.ndarray, float]:
    model.eval()
    preds_all: List[int] = []
    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(1)

        preds_all.extend(preds.detach().cpu().tolist())
        correct += (preds == y).sum().item()
        total += y.numel()

    acc = 100.0 * correct / total
    return np.array(preds_all, dtype=np.int64), float(acc)


def save_training_curve(history_lines: List[str]) -> None:
    epochs: List[int] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    for line in history_lines[1:]:
        epoch_str, train_str, val_str = line.split(",")
        epochs.append(int(epoch_str))
        train_accs.append(float(train_str) * 100.0)
        val_accs.append(float(val_str) * 100.0)

    best_epoch = epochs[int(np.argmax(val_accs))]
    best_val = max(val_accs)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker="o", linewidth=2, color="royalblue", label="Training Accuracy")
    plt.plot(epochs, val_accs, marker="o", linewidth=2, color="darkorange", label="Validation Accuracy")

    plt.scatter(
        [best_epoch],
        [best_val],
        s=120,
        color="red",
        edgecolor="black",
        zorder=5,
        label=f"Best Val Epoch ({best_epoch})",
    )
    plt.axvline(
        x=best_epoch,
        linestyle="--",
        linewidth=1.5,
        color="red",
        alpha=0.6,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Q5 CIFAR-10 Training vs Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "q5_training_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    cfg = TrainConfig()
    seed_everything(cfg.seed)
    ensure_dirs()

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
    ])

    transform_eval = T.Compose([
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
    ])

    # Training dataset with augmentation
    train_full_aug = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "cifar_data"),
        train=True,
        download=True,
        transform=transform_train,
    )

    # Validation dataset with deterministic evaluation transform
    train_full_eval = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "cifar_data"),
        train=True,
        download=True,
        transform=transform_eval,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "cifar_data"),
        train=False,
        download=True,
        transform=transform_eval,
    )

    val_len = int(len(train_full_aug) * cfg.val_split)
    train_len = len(train_full_aug) - val_len

    all_indices = np.arange(len(train_full_aug))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_indices)

    train_indices = all_indices[:train_len]
    val_indices = all_indices[train_len:]

    train_set = Subset(train_full_aug, train_indices.tolist())
    val_set = Subset(train_full_eval, val_indices.tolist())

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = PatchMLP(
        patch_size=cfg.patch_size,
        patch_embed_dim=cfg.patch_embed_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val = 0.0
    best_train = 0.0
    best_state = None
    history_lines = ["epoch,train_acc,val_acc"]

    for epoch in range(1, cfg.epochs + 1):
        train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            opt=opt,
            device=cfg.device,
            label_smoothing=cfg.label_smoothing,
        )
        val_acc = eval_model(model, val_loader, cfg.device)

        history_lines.append(f"{epoch},{train_acc:.6f},{val_acc:.6f}")

        if val_acc > best_val:
            best_val = val_acc
            best_train = train_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch:02d}/{cfg.epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    preds, test_acc_pct = test_mlp(model, test_loader, cfg.device)
    print(f"test accuracy: {test_acc_pct:.2f}%")

    npy_path = OUTPUTS_DIR / "q5_predictions.npy"
    np.save(npy_path, preds)
    print(f"saved: {npy_path}")

    history_path = OUTPUTS_DIR / "q5_history.csv"
    history_path.write_text("\n".join(history_lines) + "\n", encoding="utf-8")
    print(f"saved: {history_path}")

    curve_path = OUTPUTS_DIR / "q5_training_curve.png"
    save_training_curve(history_lines)
    print(f"saved: {curve_path}")

    metrics_path = OUTPUTS_DIR / "q5_metrics.txt"
    metrics_text = (
        "Q5 CIFAR-10 MLP results\n"
        f"device={cfg.device}\n"
        f"patch_size={cfg.patch_size}\n"
        f"patch_embed_dim={cfg.patch_embed_dim}\n"
        f"hidden_dims={cfg.hidden_dims}\n"
        f"dropout={cfg.dropout}\n"
        f"batch_size={cfg.batch_size}\n"
        f"lr={cfg.lr}\n"
        f"epochs={cfg.epochs}\n"
        f"weight_decay={cfg.weight_decay}\n"
        f"label_smoothing={cfg.label_smoothing}\n"
        f"best_train_acc={best_train * 100:.2f}%\n"
        f"best_val_acc={best_val * 100:.2f}%\n"
        f"test_acc={test_acc_pct:.2f}%\n"
        f"predictions_file={npy_path.name}\n"
        f"history_file={history_path.name}\n"
        f"curve_file={curve_path.name}\n"
        f"metrics_file={metrics_path.name}\n"
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()