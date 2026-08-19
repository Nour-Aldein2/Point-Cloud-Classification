from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import Config
from data_utils import PointCloudData
from model import PointNetBaseline


def one_epoch(model, optimizer, loss_fcn, train_loader, val_loader, device):
    train_loss = 0.0
    train_correct = 0
    train_samples = 0
    model.train()
    for batch in tqdm(train_loader, desc="Training", ncols=80, leave=False):
        points = batch["point_cloud"].to(device)
        labels = batch["category"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(points)
        loss = loss_fcn(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        train_loss += loss.item() * batch_size
        train_correct += (logits.argmax(dim=1) == labels).sum().item()

        train_samples += batch_size

    train_loss /= train_samples
    train_acc = train_correct / train_samples

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Validating", ncols=80, leave=False):
            points = batch["point_cloud"].to(device)
            labels = batch["category"].to(device)

            logits = model(points)
            loss = loss_fcn(logits, labels)

            batch_size = labels.size(0)

            val_loss += loss.item() * batch_size
            val_correct += (logits.argmax(dim=1) == labels).sum().item()

            val_samples += batch_size

    val_loss /= val_samples
    val_acc = val_correct / val_samples

    return train_loss, train_acc, val_loss, val_acc




if __name__ == "__main__":
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    train_dataset = PointCloudData(root_dir=cfg.data.path,
                                   num_points=cfg.data.num_points,
                                   split_name="train",
                                   seed=cfg.data.seed)
    val_dataset = PointCloudData(root_dir=cfg.data.path,
                                  num_points=cfg.data.num_points,
                                  split_name="val",
                                  seed=cfg.data.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    baseline_model = PointNetBaseline(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=cfg.training.learning_rate)

    checkpoint_dir = cfg.training.save_path
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    es_counter = 0
    for epoch in range(cfg.training.epochs):

        train_loss, train_acc, val_loss, val_acc = one_epoch(
            model=baseline_model,
            optimizer=optimizer,
            loss_fcn=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.2%} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            es_counter = 0
            print(f" ✅ New best model found val_loss={val_loss:.4f} (prev ={best_val_loss:.4f})")
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": baseline_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint_dir / "best_model.pt",
            )
        else:
            es_counter += 1

        if es_counter >= cfg.training.es_patience:
            print("Early stopping triggered!")
            break