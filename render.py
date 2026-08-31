"""Rendering Polish text with PIL: words, lines and whole pages."""

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts", str(Path.home() / ".fonts")]


def find_fonts(search_dirs=None, require_chars="ąćęłńóśźż"):
    """All installed .ttf faces that have a glyph for every Polish letter."""
    from fontTools.ttLib import TTFont

    found = []
    for search_dir in search_dirs or FONT_DIRS:
        root = Path(search_dir).expanduser()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.ttf")):
            try:
                with TTFont(str(path), fontNumber=0, lazy=True) as face:
                    covered = set()
                    for table in face["cmap"].tables:
                        covered.update(chr(cp) for cp in table.cmap)
            except Exception:
                continue
            if all(char in covered for char in require_chars):
                found.append(str(path))
    return found


@lru_cache(maxsize=256)
def load_font(font_path, size_px):
    """A PIL font, read from disk once per (file, size) pair."""
    return ImageFont.truetype(font_path, size_px)


def render_word(text, font_path, size_px=15, padding=2, background=255, ink=0):
    """One word as a greyscale (H, W) uint8 array."""
    import numpy as np

    if not text:
        raise ValueError("cannot render an empty string")
    font = load_font(font_path, size_px)
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = measure.textbbox((0, 0), text, font=font)
    canvas = Image.new("L", (x1 - x0 + 2 * padding, y1 - y0 + 2 * padding), color=background)
    draw = ImageDraw.Draw(canvas)
    draw.text((padding - x0, padding - y0), text, font=font, fill=ink)
    return np.asarray(canvas, dtype=np.uint8)


def word_boxes(text, font, origin):
    """Split a rendered line into words and give each one a box in page pixels.

    Boxes are exact horizontally - the offsets come from the same font metrics
    the text was drawn with - and are the full line height vertically, ascender
    to descender, rather than a tight fit around the glyphs.
    """
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    x, y = origin
    ascent, descent = font.getmetrics()
    boxes = []
    cursor = 0
    for token in text.split(" "):
        prefix = text[:cursor]
        offset = measure.textlength(prefix, font=font) if prefix else 0.0
        advance = measure.textlength(token, font=font)
        if token:
            boxes.append((token, (x + offset, y, x + offset + advance, y + ascent + descent)))
        cursor += len(token) + 1
    return boxes


def wrap(text, font, max_width):
    """Greedy word wrap measured with the real font metrics."""
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines, current = [], []
    for token in text.split():
        candidate = " ".join([*current, token])
        if current and measure.textlength(candidate, font=font) > max_width:
            lines.append(" ".join(current))
            current = [token]
        else:
            current.append(token)
    if current:
        lines.append(" ".join(current))
    return lines


def normalise_bbox(box, width, height):
    """Pixel box -> fractions of the page size, clipped to the page."""
    left, top, right, bottom = box
    return (
        min(max(left / width, 0.0), 1.0),
        min(max(top / height, 0.0), 1.0),
        min(max(right / width, 0.0), 1.0),
        min(max(bottom / height, 0.0), 1.0),
    )
