"""Train and evaluate the transformer-CNN hate-speech classifier.

Examples
--------
Spanish-only, XLM-RoBERTa vs. BETO baseline::

    python train.py --backbone xlm-roberta-base --train data/train_es.csv \
        --val data/val_es.csv --test data/test_es.csv --epochs 4
    python train.py --backbone beto --train data/train_es.csv \
        --val data/val_es.csv --test data/test_es.csv --epochs 4

Multilingual fine-tuning (pure vs. translated) — pass several training files
and evaluate on each language's test set::

    python train.py --backbone xlm-roberta-base \
        --train data/train_es.csv data/train_turkish.csv data/train_french.csv \
        --val data/val_es.csv --test data/test_es.csv --epochs 1
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.utils import shuffle

from data import get_tokenizer, load_csvs, make_loader
from model import TransformerCNN

SEED = 1234


def set_seed(seed: int = SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one pass over ``loader``; train if an optimizer is given, else eval."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    probs, targets = [], []

    with torch.set_grad_enabled(is_train):
        for input_ids, attention_masks, labels in loader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device).float().reshape(-1, 1)

            output = model(input_ids, attention_masks)
            loss = criterion(output, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            probs.append(output.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    return total_loss / len(loader), np.vstack(probs), np.vstack(targets)


def report(probs, targets, title):
    preds = (probs.ravel() > 0.5).astype(int)
    y_true = targets.ravel().astype(int)
    print(f"\n=== {title} ===")
    print(classification_report(y_true, preds, digits=3))
    print(f"F1 (toxic): {f1_score(y_true, preds):.3f}")
    print(f"Recall (toxic): {recall_score(y_true, preds):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="xlm-roberta-base", choices=["xlm-roberta-base", "beto"])
    parser.add_argument("--train", nargs="+", required=True, help="one or more training CSV files")
    parser.add_argument("--val", required=True, help="validation CSV file")
    parser.add_argument("--test", required=True, help="test CSV file")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3, help="early-stopping patience")
    parser.add_argument("--checkpoint", default="best-model.pt")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(args.backbone)
    train_df = shuffle(load_csvs(args.train), random_state=SEED)
    train_loader = make_loader(train_df, tokenizer, args.batch_size, shuffle=True)
    val_loader = make_loader(load_csvs(args.val), tokenizer, args.batch_size, shuffle=False)
    test_loader = make_loader(load_csvs(args.test), tokenizer, args.batch_size, shuffle=False)

    model = TransformerCNN(args.backbone).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_probs, val_targets = run_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    report(val_probs, val_targets, "Validation (last epoch)")

    # Evaluate the best checkpoint on the held-out test set.
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    _, test_probs, test_targets = run_epoch(model, test_loader, criterion, device)
    report(test_probs, test_targets, "Test (best checkpoint)")


if __name__ == "__main__":
    main()
