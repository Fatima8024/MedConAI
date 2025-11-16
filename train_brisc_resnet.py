# train_brisc_resnet.py
# ---------------------------------------------------------
# Train a BRISC2025 classifier using ResNet18.
#
# Features:
# - Uses BRISC2025Loader + configs/brisc.yaml  (same as app)
# - Logs per-class accuracy on validation
# - Saves:
#     brisc_exports/brisc_resnet_last.pth      (for resume)
#     brisc_exports/brisc_resnet_epochXX.pth   (per-epoch model_state)
#     brisc_exports/best_cls.pth               (best model_state)
#     brisc_exports/brisc_classifier.pt        (TorchScript of best model)
# - Auto-resume:
#     Each run trains for +EPOCHS_PER_RUN epochs from last checkpoint.
#
# Educational use only. Not a medical device.
# ---------------------------------------------------------

import os
import io
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from tqdm.auto import tqdm

from utils.config import load_cfg
from data.loader_registry import get_loader


# ---------------- Paths / Config ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(BASE_DIR, "brisc2025", "classification_task")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "test")   # use test as val for now

EXPORT_DIR = os.path.join(BASE_DIR, "brisc_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

CFG = load_cfg(os.path.join(BASE_DIR, "configs", "brisc.yaml"))
LABELS: List[str] = list(CFG.inference.labels)
NUM_CLASSES = len(LABELS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how many *additional* epochs per run
EPOCHS_PER_RUN = 4

LAST_CKPT_PATH = os.path.join(EXPORT_DIR, "brisc_resnet_last.pth")
BEST_CKPT_PATH = os.path.join(EXPORT_DIR, "best_cls.pth")


# ---------------- Loader (BRISC preproc) ----------------

Loader = get_loader("brisc2025")
_preproc = Loader(CFG.preproc)


class BriscDataset(Dataset):
    def __init__(self, root: str, labels: List[str]):
        self.samples: List[Tuple[str, int]] = []
        self.labels = labels
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

        for lbl in labels:
            cls_dir = os.path.join(root, lbl)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Missing class dir: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if (
                    os.path.isfile(fpath)
                    and fname.lower().endswith((".jpg", ".jpeg", ".png"))
                ):
                    self.samples.append((fpath, label_to_idx[lbl]))

        print(f"[INIT] {root} -> {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arr = _preproc.decode_upload(buf.getvalue())
        x = _preproc.to_tensor(arr)
        x = _preproc.normalize(x)

        return x, label


# ---------------- Model builder ----------------

def build_model(num_classes: int) -> nn.Module:
    # Try modern API first, fallback if older torchvision
    try:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        m = models.resnet18(pretrained=True)

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ---------------- Evaluation (with per-class stats) ----------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> float:
    model.eval()
    num_classes = NUM_CLASSES
    total = 0
    correct = 0
    class_total = [0] * num_classes
    class_correct = [0] * num_classes

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"{desc} [eval]", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            matches = (preds == y)
            correct += matches.sum().item()
            total += y.numel()

            for c in range(num_classes):
                mask = (y == c)
                ct = mask.sum().item()
                if ct > 0:
                    class_total[c] += ct
                    class_correct[c] += ((preds == c) & mask).sum().item()

    overall_acc = correct / max(1, total)
    print(f"[{desc}] overall_acc = {overall_acc:.4f} ({correct}/{total})")
    for i, lbl in enumerate(LABELS):
        if class_total[i] > 0:
            acc_i = class_correct[i] / class_total[i]
            print(
                f"[{desc}] {lbl:10s}: {acc_i:.4f} "
                f"({class_correct[i]}/{class_total[i]})"
            )
        else:
            print(f"[{desc}] {lbl:10s}: N/A (0 samples)")
    return overall_acc


# ---------------- Checkpoint helpers ----------------

def save_epoch_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_state):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
        "best_state": best_state,
        "labels": LABELS,
    }
    torch.save(ckpt, LAST_CKPT_PATH)
    epoch_ckpt_path = os.path.join(EXPORT_DIR, f"brisc_resnet_epoch{epoch:02d}.pth")
    torch.save(model.state_dict(), epoch_ckpt_path)
    print(f"[CKPT] Saved last checkpoint -> {LAST_CKPT_PATH}")
    print(f"[CKPT] Saved epoch weights  -> {epoch_ckpt_path}")


def load_checkpoint_if_exists(model, optimizer, scheduler):
    if not os.path.exists(LAST_CKPT_PATH):
        print("[CKPT] No existing checkpoint found. Starting fresh.")
        return 0, 0.0, None  # start_epoch, best_acc, best_state

    ckpt = torch.load(LAST_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_acc", 0.0)
    best_state = ckpt.get("best_state", None)

    print(
        f"[CKPT] Resumed from epoch {start_epoch} "
        f"with best_acc={best_acc:.4f}"
    )
    return start_epoch, best_acc, best_state


# ---------------- Main training ----------------

def main():
    # Datasets
    train_ds = BriscDataset(TRAIN_DIR, LABELS)
    val_ds   = BriscDataset(VAL_DIR, LABELS)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/val datasets are empty or paths incorrect.")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    # Model, loss, opt
    model = build_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Resume if possible
    start_epoch, best_acc, best_state = load_checkpoint_if_exists(
        model, optimizer, scheduler
    )

    # If we resumed and have a best_state, keep it; otherwise use current
    if best_state is None:
        best_state = model.state_dict()

    # Training loop: +EPOCHS_PER_RUN epochs from where we left off
    end_epoch = start_epoch + EPOCHS_PER_RUN
    print(
        f"[TRAIN] Running epochs {start_epoch + 1} to {end_epoch} "
        f"(EPOCHS_PER_RUN={EPOCHS_PER_RUN})"
    )

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(1, len(train_ds))
        print(f"[Epoch {epoch}] train_loss = {train_loss:.4f}")

        # Validation with per-class accuracy
        val_acc = evaluate(model, val_loader, DEVICE, desc=f"Epoch {epoch} VAL")

        # Step LR scheduler
        scheduler.step(val_acc)

        # Track and save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, BEST_CKPT_PATH)
            print(
                f"[BEST] New best_acc={best_acc:.4f} -> saved best_cls.pth at {BEST_CKPT_PATH}"
            )

        # Save "last" + per-epoch weights
        save_epoch_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_state)

    # ----- Export best model to TorchScript for app_brisc_clean.py -----

    # Load best weights (from memory or from file as backup)
    if best_state is None and os.path.exists(BEST_CKPT_PATH):
        best_state = torch.load(BEST_CKPT_PATH, map_location=DEVICE)
        print("[BEST] Loaded best_state from best_cls.pth")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Dummy input based on configured input size
    h, w = CFG.preproc.input_size
    dummy = torch.randn(1, 3, h, w).to(DEVICE)

    with torch.no_grad():
        ts_model = torch.jit.trace(model, dummy)

    ts_out_path = os.path.join(EXPORT_DIR, "brisc_classifier.pt")
    ts_model.save(ts_out_path)

    print(f"[EXPORT] TorchScript best classifier -> {ts_out_path}")
    print(f"[EXPORT] Best accuracy so far: {best_acc:.4f}")

    # List all .pth for inspection
    print("\n[FILES] .pth files in brisc_exports:")
    for fname in os.listdir(EXPORT_DIR):
        if fname.lower().endswith(".pth"):
            print("  -", os.path.join(EXPORT_DIR, fname))

    print("[DONE] Now run app_brisc_clean.py with seg-aware logic to use this model.")


if __name__ == "__main__":
    main()
