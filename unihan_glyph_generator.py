"""
Generate the Chinese glyph + definition dataset (script version of `Unihan glyph generator.ipynb`).

This step is pure CPU work (font rendering + PNG encoding), so there is nothing to put on GPUs.
Instead, rendering is parallelised across CPU cores with a process pool.

Outputs:
    <out_dir>/<HEX>.png        one 128x128 glyph per supported character
    <metadata_path>            JSON lines: {"Unicode", "Character", "caption", "file_name"}

Usage:
    python unihan_glyph_generator.py --font ./fonts/NotoSansSC-Regular.ttf --out_dir data --workers 32
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate Unihan glyph images + metadata.jsonl")
    p.add_argument("--definitions", default="Unihan-kDefinition.txt")
    p.add_argument("--font", default="./fonts/NotoSansSC-Regular.ttf")
    p.add_argument("--font_size", type=int, default=100)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--y_offset", type=int, default=-16, help="Vertical shift used by the notebook")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--metadata_path", default=None, help="Default: <out_dir>/metadata.jsonl")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--overwrite", action="store_true", help="Re-render images that already exist")
    p.add_argument("--include_unsupported", action="store_true",
                   help="Keep rows whose glyph the font cannot render (notebook behaviour). "
                        "By default they are dropped so every metadata row has an image.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Definition cleaning (identical to the notebook so captions are reproduced exactly)
# ---------------------------------------------------------------------------

def clean_definition(definition):
    # NOTE: `[/u3400-\u9FFF]` is a typo in the original notebook (should be `\u3400`). As written,
    # the class covers '/', 'u', '3', '4', '0' and the range '0'..'\u9FFF', so it removes almost every
    # parenthesised clause. It is kept unchanged so captions match the existing dataset.
    definition = re.sub(r'\(.*?[/u3400-\u9FFF]+.*?\)', '', definition)
    definition = re.sub(r',.*?[\u3400-\u9FFF]+.*?(,|$)', '', definition)
    definition = re.sub(r';.*?[\u3400-\u9FFF]+.*?(;|$)', '', definition)
    definition = re.sub(r'[\u3400-\u9FFF]', '', definition)
    definition = re.sub(r'U\+\w+', '', definition)
    definition = re.sub(r'[^\x00-\x7F]', '', definition)
    definition = re.sub(r'^[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+|[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+$', '', definition)
    definition = re.sub(r'\s[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+\s', ' ', definition)
    definition = re.sub(r'\s+', ' ', definition).strip()
    return definition


def load_definitions(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r'U\+([0-9A-F]+)\s([\u3400-\u9FFF])\tkDefinition\t(.+)', line)
            if m:
                caption = clean_definition(m.group(3))
                if caption != "":
                    rows.append({"Unicode": m.group(1), "Character": m.group(2), "caption": caption})
    return rows


# ---------------------------------------------------------------------------
# Rendering (runs in worker processes)
# ---------------------------------------------------------------------------

_W = {}  # per-worker state: font, settings, bytes of the "unsupported" glyph


def create_image(character, font, image_size, y_offset):
    image = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textbbox((0, 0), character, font=font)[2:]
    x = (image_size - text_width) / 2
    y = (image_size - text_height) / 2 + y_offset
    draw.text((x, y), character, fill='black', font=font)
    return image


def _init_worker(font_path, font_size, image_size, y_offset, out_dir, overwrite):
    font = ImageFont.truetype(font_path, font_size)
    _W.update(
        font=font, image_size=image_size, y_offset=y_offset, out_dir=out_dir, overwrite=overwrite,
        # The notebook re-rendered U+FFFD for every character; compute it once per worker instead.
        unknown=create_image("\uFFFD", font, image_size, y_offset).tobytes(),
    )


def _render(unicode_hex):
    """Returns (unicode_hex, supported)."""
    code_point = int(unicode_hex, 16)
    filename = os.path.join(_W["out_dir"], f"{code_point:04X}.png")
    img = create_image(chr(code_point), _W["font"], _W["image_size"], _W["y_offset"])
    if img.tobytes() == _W["unknown"]:
        return unicode_hex, False
    if _W["overwrite"] or not os.path.exists(filename):
        img.save(filename)
    return unicode_hex, True


def main():
    args = parse_args()
    metadata_path = args.metadata_path or os.path.join(args.out_dir, "metadata.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_definitions(args.definitions)
    print(f"{len(rows)} characters with non-empty definitions")

    supported = {}
    init_args = (args.font, args.font_size, args.image_size, args.y_offset, args.out_dir, args.overwrite)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=init_args) as ex:
        it = ex.map(_render, [r["Unicode"] for r in rows], chunksize=256)
        for unicode_hex, ok in tqdm(it, total=len(rows), desc="Rendering glyphs"):
            supported[unicode_hex] = ok

    n_ok = sum(supported.values())
    print(f"{n_ok} glyphs supported by the font, {len(rows) - n_ok} unsupported")

    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    n_written = 0
    with open(metadata_path, "w", encoding="utf-8") as f:
        for r in rows:
            if not args.include_unsupported and not supported[r["Unicode"]]:
                continue
            rec = {**r, "file_name": r["Unicode"] + ".png"}
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            n_written += 1
    print(f"Wrote {n_written} rows to {metadata_path}")


if __name__ == "__main__":
    main()
