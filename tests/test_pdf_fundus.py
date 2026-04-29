from io import BytesIO
from pathlib import Path

from PIL import Image

from vascx_models.pdf_fundus import (
    crop_bright_margins,
    decode_image_bytes,
    iter_pdfs,
    select_primary_image,
)


def test_iter_pdfs_recurses_and_skips_hidden_paths(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    hidden = tmp_path / ".hidden"
    visible.mkdir()
    hidden.mkdir()
    first = tmp_path / "a.pdf"
    second = visible / "b.pdf"
    skipped = hidden / "c.pdf"
    first.write_bytes(b"%PDF")
    second.write_bytes(b"%PDF")
    skipped.write_bytes(b"%PDF")
    (visible / "not_pdf.txt").write_text("x", encoding="utf-8")

    assert iter_pdfs(tmp_path) == [first, second]


def test_select_primary_image_prefers_color_then_area() -> None:
    candidates = [
        {"width": 100, "height": 100, "colorspace": 1, "image": b"large-gray"},
        {"width": 20, "height": 20, "cs-name": "DeviceRGB", "image": b"color"},
        {"width": 10, "height": 10, "cs-name": "DeviceRGB", "image": b"small"},
    ]

    assert select_primary_image(candidates) is candidates[1]


def test_decode_image_bytes_returns_rgb_image() -> None:
    buffer = BytesIO()
    Image.new("L", (4, 3), color=128).save(buffer, format="PNG")

    image = decode_image_bytes(buffer.getvalue())

    assert image.mode == "RGB"
    assert image.size == (4, 3)


def test_crop_bright_margins_trims_white_border() -> None:
    image = Image.new("RGB", (10, 8), color=(255, 255, 255))
    for x in range(3, 7):
        for y in range(2, 6):
            image.putpixel((x, y), (10, 20, 30))

    cropped = crop_bright_margins(image)

    assert cropped.size == (4, 4)
