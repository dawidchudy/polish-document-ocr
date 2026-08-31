"""Run layout-aware OCR over a PDF or image and save the words as JSON."""

import argparse
import json
from pathlib import Path


def prepare_recogniser(checkpoint=None, workdir=Path("out")):
    """Put the recognition weights on disk next to a config describing the model.

    deepdoctection builds the docTR model itself and only then loads weights into
    it, so the two have to agree on the vocabulary, or the output layer comes out
    the wrong width and loading fails on `linear.weight`. Its built-in profile
    assumes a different docTR release than the one installed here. Writing the
    model's own cfg beside the checkpoint settles it, because the config and the
    weights are then produced by the same object.
    """
    import torch
    from doctr.datasets import VOCABS
    from doctr.models import recognition

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        model = recognition.crnn_vgg16_bn(pretrained=True)
        checkpoint = workdir / "stock-recogniser.pt"
        torch.save(model.state_dict(), checkpoint)
    else:
        checkpoint = Path(checkpoint)
        model = recognition.crnn_vgg16_bn(pretrained=False, vocab=VOCABS["polish"])
    config = checkpoint.with_suffix(".json")
    config.write_text(json.dumps(model.cfg), encoding="utf-8")
    return str(checkpoint.resolve()), str(config.resolve())


def register_recogniser(checkpoint, config):
    """Announce the weights to deepdoctection before asking it to load them.

    It only loads weights it has a profile for, so pointing the config at a file
    on disk is not enough on its own. `train.py` saves the plain recognition
    state dict, which is the shape `DoctrTextRecognizer` expects.
    """
    import deepdoctection as dd

    dd.ModelCatalog.register(checkpoint, dd.ModelProfile(
        name=checkpoint,
        description="CRNN text recogniser with an explicit vocabulary",
        size=[Path(checkpoint).stat().st_size],
        tp_model=False,
        architecture="crnn_vgg16_bn",
        config=config,
        dl_library="PT",
        model_wrapper="DoctrTextRecognizer",
        categories={},
    ))
    return checkpoint


def analyse(path, checkpoint=None, workdir=Path("out")):
    """Layout detection + text recognition via deepdoctection.

    Returns a {"pages": [...]} dict; each page has its words in reading order,
    with boxes normalised to fractions of the page size.
    """
    import deepdoctection as dd

    weights, config = prepare_recogniser(checkpoint, workdir)
    overrides = [
        "USE_LAYOUT=True",
        "USE_LAYOUT_NMS=True",
        "USE_TABLE_SEGMENTATION=False",
        f"OCR.WEIGHTS.DOCTR_RECOGNITION.PT={register_recogniser(weights, config)}",
    ]

    analyzer = dd.get_dd_analyzer(config_overwrite=overrides)
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        frame = analyzer.analyze(path=str(path))
    else:
        # single images are passed as path + raw bytes (deepdoctection >= 1.2)
        frame = analyzer.analyze(path=str(path), bytes=path.read_bytes())
    frame.reset_state()

    pages = []
    for number, raw in enumerate(frame, start=1):
        width, height = float(raw.width or 1), float(raw.height or 1)
        words = []
        for word in raw.words:
            left, top, right, bottom = word.bbox
            words.append({
                "text": str(word.characters),
                "bbox": [left / width, top / height, right / width, bottom / height],
                "confidence": float(getattr(word, "score", 1.0) or 1.0),
            })
        pages.append({"page": number, "width": width, "height": height, "words": words})
    return {"pages": pages}


def main():
    """Run the analyser over one document and write its words to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--checkpoint", default=None,
                        help="recogniser from train.py; omit to use the stock model")
    parser.add_argument("--out", type=Path, default=Path("out/ocr.json"))
    args = parser.parse_args()

    result = analyse(args.document, args.checkpoint, args.out.parent)
    n_words = sum(len(page["words"]) for page in result["pages"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(result['pages'])} pages, {n_words} words -> {args.out}")


if __name__ == "__main__":
    main()
