"""Build a synthetic OCR training set: render every word of a word list in every font."""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from render import find_fonts, render_word


def unique_names(font_paths):
    """A distinct directory name per font file.

    Font stems are not unique - `liberation/` and `liberation2/` ship the same
    file names - and identical stems would overwrite each other's crops while
    both kept a row in labels.csv.
    """
    names = {}
    for path in font_paths:
        path = Path(path)
        name = path.stem
        if name in names.values():
            name = f"{path.parent.name}-{path.stem}"
        while name in names.values():
            name = f"{name}-2"
        names[str(path)] = name
    return names


def read_words(path, limit=None, seed=0):
    """One word per line, blanks dropped, optionally sampled down to `limit`.

    Sampled rather than truncated: a Polish word list is alphabetical, so the
    first few hundred lines are all `aa`, `aaa`, `abaj` and share both their
    letters and their length. A model trained on that has not seen the language.
    """
    words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit and limit < len(words):
        words = random.Random(seed).sample(words, limit)
    return sorted(words)


def case_variants(word):
    """The same word lowercase, capitalised and in capitals.

    Documents use all three: running text is lowercase, given names are
    capitalised, and surnames in a form are usually shouted in capitals. A
    recogniser trained on one of them alone loses the other two.
    """
    return list(dict.fromkeys([word.lower(), word.lower().capitalize(), word.upper()]))


def field_tokens(rng, count):
    """Strings shaped like the values a form holds, which a dictionary has none of.

    Identifiers, postal codes and house numbers are most of what anyone wants to
    read off a document, and a model trained only on words forgets how to write
    a digit at all.
    """
    tokens = set()
    while len(tokens) < count:
        kind = rng.randrange(5)
        if kind == 0:
            tokens.add("".join(rng.choice("0123456789") for _ in range(11)))
        elif kind == 1:
            tokens.add("".join(rng.choices("ABCDEFGHIJKLMNOPRSTUWXYZ", k=3))
                       + f"{rng.randrange(10 ** 6):06d}")
        elif kind == 2:
            tokens.add(f"{rng.randint(0, 99):02d}-{rng.randint(100, 999)}")
        elif kind == 3:
            tokens.add(str(rng.randint(1, 999)))
        else:
            tokens.add(f"{rng.randint(1, 99)}/{rng.randint(1, 99)}")
    return sorted(tokens)


def main():
    """Render every token of the list in every usable font, with a labels.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words-file", type=Path, required=True,
                        help="one Polish word per line (e.g. from sjp.pl)")
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    parser.add_argument("--limit", type=int, default=None, help="cap on distinct words")
    parser.add_argument("--seed", type=int, default=0, help="which words get sampled")
    parser.add_argument("--field-tokens", type=int, default=400,
                        help="identifiers, postal codes and house numbers to mix in")
    parser.add_argument("--max-fonts", type=int, default=None,
                        help="cap on typefaces; the corpus is words x fonts, so this "
                             "is the knob that decides whether it fits on disk")
    args = parser.parse_args()

    fonts = find_fonts()
    if not fonts:
        raise SystemExit("no usable fonts found; install ttf fonts with Polish glyphs")
    if args.max_fonts:
        # Sampled, not the first n, so the faces are not all one family.
        fonts = sorted(random.Random(args.seed).sample(fonts, min(args.max_fonts, len(fonts))))
    names = unique_names(fonts)
    print(f"{len(fonts)} usable fonts")

    rng = random.Random(args.seed)
    words = read_words(args.words_file, args.limit, args.seed)
    # Punctuation rides along with the word it touches, because the detector
    # hands over `Nazwisko:` as a single token, colon included.
    tokens = [variant for word in words for variant in case_variants(word)]
    tokens += [word + mark for word in rng.sample(words, max(1, len(words) // 10))
               for mark in (":", ",")]
    tokens += field_tokens(rng, args.field_tokens)
    tokens = sorted(set(tokens))
    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow("filename text font".split())
        for position, font_path in enumerate(fonts, start=1):
            font_name = names[font_path]
            font_dir = crops_dir / font_name
            font_dir.mkdir(exist_ok=True)
            size_px = rng.randint(13, 20)
            for index, token in enumerate(tokens):
                image = render_word(token, font_path, size_px=size_px)
                Image.fromarray(image).save(font_dir / f"{index:06d}.png")
                writer.writerow([f"crops/{font_name}/{index:06d}.png", token, font_name])
            if position % 25 == 0 or position == len(fonts):
                print(f"  {position}/{len(fonts)} fonts")

    print(f"wrote {len(tokens) * len(fonts)} crops of {len(tokens)} tokens to {args.out}")


if __name__ == "__main__":
    main()
