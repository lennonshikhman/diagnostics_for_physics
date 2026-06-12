from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .normalization import Normalizer

def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: Normalizer,
    epochs: int,
    lr: float,
    device: torch.device,
    out_ckpt: Path,
) -> Dict[str, Any]:
    model.to(device)
    norm = normalizer.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = -1
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()
            tau = batch["tau"].to(device).float()
            xn = norm.encode(x)
            yn = norm.encode(y)
            pred = model(xn, tau)
            loss = F.mse_loss(pred, yn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device).float()
                y = batch["y"].to(device).float()
                tau = batch["tau"].to(device).float()
                pred = model(norm.encode(x), tau)
                val_losses.append(float(F.mse_loss(pred, norm.encode(y)).detach().cpu()))
        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        hist.append({"epoch": epoch, "train_mse": tr, "val_mse": va})
        if va < best_val:
            best_val = va
            best_epoch = epoch
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "best_val": best_val, "best_epoch": best_epoch}, out_ckpt)
        print(f"    epoch {epoch:04d}/{epochs:04d} train_mse={tr:.4e} val_mse={va:.4e}")
    if out_ckpt.exists():
        ckpt = torch.load(out_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    return {"best_val": best_val, "best_epoch": best_epoch, "history": hist}


