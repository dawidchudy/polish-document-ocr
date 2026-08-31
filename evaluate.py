"""Measure recognition: on held-out words, and on a rendered page."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import CropDataset, collate, read_labels, split_by_word

FOLD = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})


def fold(text):
    """Strip Polish diacritics, so `Kraków` and `Krakow` compare equal."""
    return text.translate(FOLD)


def load_model(checkpoint, device):
    """The stock recogniser, or ours if a checkpoint is given.

    Stock means docTR's published weights and its own vocabulary, which has no
    Polish letters in it at all. That is the thing being measured.
    """
    from doctr.datasets import VOCABS
    from doctr.models import recognition

    if checkpoint is None:
        model = recognition.crnn_vgg16_bn(pretrained=True)
    else:
        model = recognition.crnn_vgg16_bn(pretrained=False, vocab=VOCABS["polish"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def predict(model, loader, device):
    """Run the recogniser over a loader and pair each prediction with its truth."""
    pairs = []
    with torch.no_grad():
        for images, texts in loader:
            out = model(images.to(device), return_preds=True)
            for (prediction, _), truth in zip(out["preds"], texts):
                pairs.append((prediction, truth))
    return pairs


def score(pairs):
    """Exact and diacritics-folded agreement, plus how many differ only by accents."""
    exact = sum(prediction == truth for prediction, truth in pairs)
    folded = sum(fold(prediction) == fold(truth) for prediction, truth in pairs)
    total = max(len(pairs), 1)
    return {
        "exact": exact / total,
        "folded": folded / total,
        "accents_only": (folded - exact) / total,
        "n": len(pairs),
    }


def score_page(ocr_json, truth_json):
    """How many of a page's words came back, against the boxes it was drawn from.

    Crops are the clean measurement. A page is the honest one, because the
    recogniser has to read whatever the detector hands it: capitals, digits and
    a colon stuck to the end of a label. Counted as a multiset, so word order
    and reading order do not enter into it.
    """
    def words(path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return [word["text"] for page in payload["pages"] for word in page["words"]]

    read, truth = Counter(words(ocr_json)), Counter(words(truth_json))
    total = max(sum(truth.values()), 1)
    exact = sum((read & truth).values())
    folded_read = Counter(fold(text) for text in read.elements())
    folded_truth = Counter(fold(text) for text in truth.elements())
    folded = sum((folded_read & folded_truth).values())
    return {"exact": exact / total, "folded": folded / total,
            "accents_only": (folded - exact) / total, "n": total}


def report(name, result):
    """One line per model, as percentages."""
    print(f"{name:22s} exact {result['exact']:6.1%}   "
          f"diacritics folded {result['folded']:6.1%}   "
          f"differ only by accents {result['accents_only']:6.1%}")


def main():
    """Evaluate one or both models on the words held out of training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="fine-tuned recogniser from train.py")
    parser.add_argument("--no-stock", action="store_true",
                        help="skip the baseline and only score the checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="cap on validation crops")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=8,
                        help="how many disagreements to print")
    parser.add_argument("--page", type=Path, default=None,
                        help="an ocr.py JSON to score against --truth instead")
    parser.add_argument("--truth", type=Path, default=None,
                        help="the ground_truth.json synthdoc.py wrote for that page")
    args = parser.parse_args()

    if args.page:
        if not args.truth:
            raise SystemExit("--page needs --truth, the ground_truth.json of that page")
        report(args.page.stem, score_page(args.page, args.truth))
        return

    device = torch.device(args.device)
    rows = read_labels(args.corpus)
    # The same split as training, from the same seed, so these words are unseen.
    _, val_rows = split_by_word(rows, args.val_frac, args.seed)
    if args.limit:
        val_rows = val_rows[:args.limit]
    print(f"{len(val_rows)} held-out crops, "
          f"{len({row[1] for row in val_rows})} distinct words\n")

    models = []
    if not args.no_stock:
        models.append(("stock docTR", None))
    if args.checkpoint:
        models.append(("fine-tuned (Polish)", args.checkpoint))
    if not models:
        raise SystemExit("nothing to evaluate: pass --checkpoint, or drop --no-stock")

    last_pairs = None
    for name, checkpoint in models:
        model = load_model(checkpoint, device)
        height, width = model.cfg["input_shape"][1:]
        loader = DataLoader(
            CropDataset(args.corpus, val_rows, model.cfg["mean"], model.cfg["std"],
                        height, width),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate)
        pairs = predict(model, loader, device)
        report(name, score(pairs))
        last_pairs = (name, pairs)

    if args.examples and last_pairs:
        name, pairs = last_pairs
        wrong = [(p, t) for p, t in pairs if p != t][:args.examples]
        if wrong:
            print(f"\nfirst {len(wrong)} disagreements from {name}:")
            for prediction, truth in wrong:
                note = "  (accents only)" if fold(prediction) == fold(truth) else ""
                print(f"  read {prediction!r:24s} for {truth!r}{note}")


if __name__ == "__main__":
    main()
