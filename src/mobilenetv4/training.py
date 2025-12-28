from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm  # Auto-detects notebook vs terminal
except ImportError:
    from tqdm import tqdm

from src.mobilenetv4.ft_utils import generate_ft_batch


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    precision_per_class: np.ndarray
    recall_per_class: np.ndarray
    f1_per_class: np.ndarray
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ft_weight: float = 0.5,
    save_callback: Optional[Callable[[int], None]] = None,
    save_interval: int = 100,  # Save every N batches
    start_batch_idx: int = 0,  # Resume from this batch (for mid-epoch resume)
) -> Tuple[
    float, float, float, float
]:  # Returns: (loss, accuracy, classification_loss, fourier_loss)
    model.train()

    running_loss = 0.0
    running_classification_loss = 0.0
    running_fourier_loss = 0.0
    correct = 0
    total = 0

    fourier_criterion = nn.MSELoss()

    if start_batch_idx > 0:
        tqdm.write(f"Resuming from batch {start_batch_idx + 1}/{len(loader)}")
        loader_iter = iter(loader)
        for _ in range(start_batch_idx):
            try:
                next(loader_iter)
            except StopIteration:
                tqdm.write("Warning: All batches already processed. Epoch complete.")
                return 0.0, 0.0
    else:
        loader_iter = iter(loader)

    disable_pbar = os.environ.get("TQDM_DISABLE", "0") == "1"

    pbar = tqdm(
        total=len(loader),
        initial=start_batch_idx,
        desc="Training",
        leave=True,
        file=sys.stderr,
        ncols=120,
        unit="batch",
        dynamic_ncols=False,
        mininterval=0.5,
        miniters=1,
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}] {postfix}",
        disable=disable_pbar,
    )

    batch_idx = start_batch_idx
    for images, labels in loader_iter:
        actual_batch_idx = batch_idx
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs, fourier_outputs = model(images)
        fourier_map_size = (fourier_outputs.shape[2], fourier_outputs.shape[3])
        fourier_targets = generate_ft_batch(images, fourier_map_size).to(device)

        classification_loss = criterion(outputs, labels)
        fourier_loss = fourier_criterion(fourier_outputs, fourier_targets)
        loss = (1.0 - ft_weight) * classification_loss + ft_weight * fourier_loss

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        running_classification_loss += float(classification_loss.item())
        running_fourier_loss += float(fourier_loss.item())
        _, predicted = outputs.max(1)
        total += int(labels.size(0))
        correct += int(predicted.eq(labels).sum().item())

        current_accuracy = 100.0 * correct / max(total, 1)
        processed_batches = batch_idx - start_batch_idx + 1
        current_loss = running_loss / processed_batches
        current_classification_loss = running_classification_loss / processed_batches
        current_fourier_loss = running_fourier_loss / processed_batches

        pbar.set_postfix(
            {
                "loss": f"{current_loss:.4f}",
                "cls": f"{current_classification_loss:.4f}",
                "ft": f"{current_fourier_loss:.4f}",
                "acc": f"{current_accuracy:.2f}%",
                "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
            }
        )
        pbar.update(1)

        if (
            save_callback
            and save_interval > 0
            and (actual_batch_idx + 1) % save_interval == 0
        ):
            save_callback(actual_batch_idx + 1)

        batch_idx += 1

    pbar.close()

    processed_batches = batch_idx - start_batch_idx
    avg_loss = running_loss / max(processed_batches, 1)
    avg_classification_loss = running_classification_loss / max(processed_batches, 1)
    avg_fourier_loss = running_fourier_loss / max(processed_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)

    return avg_loss, accuracy, avg_classification_loss, avg_fourier_loss


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> EvalMetrics:
    model.eval()

    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        disable_pbar = os.environ.get("TQDM_DISABLE", "0") == "1"

        pbar = tqdm(
            loader,
            desc="Evaluating",
            leave=True,
            file=sys.stderr,
            ncols=120,
            unit="batch",
            dynamic_ncols=False,
            mininterval=0.5,
            miniters=1,
            bar_format="{desc}: {percentage:3.0f}% ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}] {postfix}",
            disable=disable_pbar,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item())

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            current_loss = running_loss / (batch_idx + 1)
            current_accuracy = (
                accuracy_score(np.asarray(all_labels), np.asarray(all_preds)) * 100.0
                if all_preds
                else 0.0
            )

            pbar.set_postfix(
                {
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_accuracy:.2f}%",
                }
            )

    pbar.close()

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    if y_true.size == 0:
        return EvalMetrics(
            loss=0.0,
            accuracy=0.0,
            precision_per_class=np.zeros(num_classes),
            recall_per_class=np.zeros(num_classes),
            f1_per_class=np.zeros(num_classes),
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_precision=0.0,
            weighted_recall=0.0,
            weighted_f1=0.0,
        )

    accuracy = accuracy_score(y_true, y_pred) * 100.0
    precision = precision_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )

    macro_precision = (
        precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    )
    macro_recall = (
        recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    weighted_precision = (
        precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    )
    weighted_recall = (
        recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    return EvalMetrics(
        loss=running_loss / max(len(loader), 1),
        accuracy=accuracy,
        precision_per_class=precision * 100.0,
        recall_per_class=recall * 100.0,
        f1_per_class=f1 * 100.0,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
    )
