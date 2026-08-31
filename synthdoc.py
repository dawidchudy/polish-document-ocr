"""Generate synthetic structured Polish documents with exact ground-truth word boxes."""

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from render import load_font, normalise_bbox, wrap, word_boxes

FIELD_LABELS = {
    "surname": "Nazwisko",
    "given_name": "Imię",
    "national_id": "Identyfikator",
    "document_number": "Numer dokumentu",
    "street": "Ulica",
    "house_number": "Numer domu",
    "postal_code": "Kod pocztowy",
    "city": "Miejscowość",
    "country": "Kraj",
}

CLAUSES = (
    "Dokument został sporządzony w dwóch jednobrzmiących egzemplarzach, "
    "po jednym dla każdej ze stron.",
    "Strony potwierdzają zgodność danych zawartych w niniejszym dokumencie "
    "ze stanem faktycznym.",
    "Wszelkie zmiany wymagają formy pisemnej pod rygorem nieważności.",
)

SURNAMES = ["KOWALSKI", "NOWAK", "WIŚNIEWSKI", "WOJCIK", "KOWALCZYK", "ZIELIŃSKI"]
GIVEN_NAMES = ["Jan", "Anna", "Piotr", "Maria", "Krzysztof", "Agnieszka"]
STREETS = ["Przykładowa", "Testowa", "Kwiatowa", "Polna", "Miodowa"]
CITIES = ["Warszawa", "Kraków", "Łódź", "Wrocław", "Gdańsk", "Poznań"]


@dataclass
class EntityGroup:
    """One party of the document and its field values."""

    name: str
    values: dict[str, str] = field(default_factory=dict)


def make_group(rng, name):
    """Fabricate one record. Identifiers are shaped like real ones, all fake."""
    return EntityGroup(name, {
        "surname": rng.choice(SURNAMES),
        "given_name": rng.choice(GIVEN_NAMES),
        "national_id": "".join(rng.choice("0123456789") for _ in range(11)),
        "document_number": f"{''.join(rng.choices('ABCDEF', k=3))}{rng.randrange(10**6):06d}",
        "street": rng.choice(STREETS),
        "house_number": str(rng.randint(1, 99)),
        "postal_code": f"{rng.randint(0, 99):02d}-{rng.randint(100, 999)}",
        "city": rng.choice(CITIES),
        "country": "Polska",
    })


def page_size(dpi=150):
    """A4 in pixels at this dpi."""
    return round(8.268 * dpi), round(11.693 * dpi)


def render_page(groups, font_path, dpi=150):
    """Draw a structured page; returns the image plus every word box in it."""
    width, height = page_size(dpi)
    margin = round(90 * dpi / 150)
    body = load_font(font_path, round(11 * dpi / 72))
    heading = load_font(font_path, round(18 * dpi / 72))

    lines = []
    y = float(margin)

    def emit(text, font, indent=0.0):
        """Queue one line at the current height, then advance past it."""
        nonlocal y
        if text.strip():
            lines.append((text, font, margin + indent, y))
        y += sum(font.getmetrics()) * 1.6

    emit("DOKUMENT STRUKTURYZOWANY", heading)
    emit("", body)
    for index, group in enumerate(groups, start=1):
        emit(f"Strona {index}", body)
        for key, value in group.values.items():
            emit(f"{FIELD_LABELS.get(key, key)}: {value}", body, indent=24)
        emit("", body)

    emit("Postanowienia ogólne", body)
    for number, clause in enumerate(CLAUSES, start=1):
        for i, line in enumerate(wrap(f"{number}. {clause}", body, width - 2 * margin)):
            emit(line, body, indent=0 if i == 0 else 18)
    emit("", body)
    for group in groups:
        emit(f"Podpis: {group.values['given_name']} {group.values['surname']}", body)

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    words = []
    for text, font, x, top in lines:
        draw.text((x, top), text, font=font, fill=0)
        for token, box in word_boxes(text, font, (x, top)):
            words.append({
                "text": token,
                "bbox": [round(v, 6) for v in normalise_bbox(box, width, height)],
                "confidence": 1.0,
            })

    return image, {"pages": [{"page": 1, "width": width, "height": height, "words": words}]}


def main():
    """Render one page and write the image beside its exact word boxes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", default=None, help="a .ttf face with Polish glyphs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    from render import find_fonts

    font = args.font or find_fonts()[0]
    rng = random.Random(args.seed)
    groups = [make_group(rng, "party_a"), make_group(rng, "party_b")]
    image, ground_truth = render_page(groups, font)
    args.out.mkdir(parents=True, exist_ok=True)
    image.save(args.out / "page.png")
    (args.out / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out / 'page.png'} and {args.out / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
