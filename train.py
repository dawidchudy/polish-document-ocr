"""Fine-tune a docTR text recogniser on the rendered Polish corpus."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def read_labels(corpus):
    """Every row of the corpus labels.csv as (relative crop path, text, font)."""
    rows = []
    with open(Path(corpus) / "labels.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["filename"], row["text"], row.get("font", "")))
    if not rows:
        raise SystemExit(f"no rows in {corpus}/labels.csv; run make_corpus.py first")
    return rows


def split_by_word(rows, val_frac=0.1, seed=0):
    """Hold out whole words rather than whole crops.

    The same word is rendered once per font, so splitting crop by crop would put
    `umowa` in training in one typeface and in validation in another. The model
    would then be scored on strings it has already learned, and the number would
    mean nothing.
    """
    words = sorted({text for _, text, _ in rows})
    rng = random.Random(seed)
    rng.shuffle(words)
    cut = max(1, int(len(words) * val_frac))
    held_out = set(words[:cut])
    train = [row for row in rows if row[1] not in held_out]
    val = [row for row in rows if row[1] in held_out]
    return train, val


class CropDataset(Dataset):
    """Rendered word crops, letterboxed into the recogniser's input size."""

    def __init__(self, corpus, rows, mean, std, height, width):
        self.corpus = Path(corpus)
        self.rows = rows
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        """One (image tensor, text) pair, padded on the right with background."""
        filename, text, _ = self.rows[index]
        image = Image.open(self.corpus / filename).convert("L")
        scale = self.height / image.height
        new_width = max(1, min(self.width, round(image.width * scale)))
        image = image.resize((new_width, self.height), Image.BILINEAR)
        # Pad with white, because the crops are dark ink on a white page and a
        # black pad would look like ink the recogniser has to explain.
        canvas = Image.new("L", (self.width, self.height), color=255)
        canvas.paste(image, (0, 0))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = np.repeat(array[None, :, :], 3, axis=0)
        return torch.from_numpy((array - self.mean) / self.std), text


def collate(batch):
    """Stack the images and keep the strings as a list, which is what docTR wants."""
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, scaler=None):
    """One pass over the loader. With an optimizer it trains, without it evaluates."""
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    for images, texts in loader:
        images = images.to(device)
        with torch.set_grad_enabled(training):
            if scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss = model(images, texts)["loss"]
            else:
                loss = model(images, texts)["loss"]
        if training:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                # A scaler that meets an overflow skips the update and lowers the
                # scale, so the schedule must not advance on a step never taken.
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                optimizer.step()
                stepped = True
            if stepped and scheduler is not None:
                scheduler.step()
        total += float(loss)
        batches += 1
    return total / max(batches, 1)


def build_model(vocab_name="polish", pretrained=True):
    """A CRNN with the Polish character set, starting from docTR's weights.

    The pretrained recogniser has a different vocabulary, so docTR keeps the
    convolutional backbone and reinitialises the output layer for ours.
    """
    from doctr.datasets import VOCABS
    from doctr.models import recognition

    return recognition.crnn_vgg16_bn(pretrained=pretrained, vocab=VOCABS[vocab_name])


def main():
    """Fine-tune the recogniser and save the best checkpoint by validation loss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--out", type=Path, default=Path("out/recogniser.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on training crops, for a quick smoke run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model().to(device)
    height, width = model.cfg["input_shape"][1:]
    mean, std = model.cfg["mean"], model.cfg["std"]

    rows = read_labels(args.corpus)
    train_rows, val_rows = split_by_word(rows, args.val_frac, args.seed)
    if args.limit:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(1, args.limit // 4)]
    print(f"{len(train_rows)} training crops, {len(val_rows)} validation crops "
          f"({len({r[1] for r in val_rows})} held-out words)")

    common = dict(mean=mean, std=std, height=height, width=width)
    train_loader = DataLoader(CropDataset(args.corpus, train_rows, **common),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(CropDataset(args.corpus, val_rows, **common),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)))
    scaler = torch.amp.GradScaler() if args.amp and device.type == "cuda" else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, scheduler, scaler)
        val_loss = run_epoch(model, val_loader, device)
        marker = ""
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), args.out)
            marker = " (saved)"
        print(f"epoch {epoch}/{args.epochs}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  {time.time() - started:.0f}s{marker}")

    print(f"best validation loss {best:.4f}, checkpoint at {args.out}")


if __name__ == "__main__":
    main()
