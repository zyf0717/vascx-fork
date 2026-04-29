from __future__ import annotations

import argparse
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def iter_pdfs(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    return sorted(
        path
        for path in source_dir.rglob("*.pdf")
        if path.is_file()
        and not any(
            part.startswith(".") for part in path.relative_to(source_dir).parts
        )
    )


def _looks_grayscale(candidate: dict) -> bool:
    colorspace = candidate.get("colorspace")
    if isinstance(colorspace, int):
        return colorspace <= 1
    if isinstance(colorspace, str):
        return "gray" in colorspace.lower()

    colorspace_name = candidate.get("cs-name")
    if isinstance(colorspace_name, str):
        return "gray" in colorspace_name.lower()
    return False


def select_primary_image(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No embedded images found in PDF page")

    selected = max(
        candidates,
        key=lambda item: (
            int(not _looks_grayscale(item)),
            int(item.get("width", 0)) * int(item.get("height", 0)),
            len(item.get("image", b"")),
        ),
    )
    LOGGER.info(
        "Selected embedded image %sx%s colorspace=%s",
        selected.get("width"),
        selected.get("height"),
        selected.get("cs-name", selected.get("colorspace")),
    )
    return selected


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise ValueError("Failed to decode embedded image bytes") from exc
    return image.convert("RGB")


def crop_bright_margins(
    image: Image.Image,
    *,
    threshold: int = 245,
    min_fraction: float = 0.01,
) -> Image.Image:
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    luminance = (
        0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    )
    mask = luminance < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        LOGGER.info(
            "No bright margins detected; keeping original image size %sx%s",
            rgb.width,
            rgb.height,
        )
        return rgb

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_width = int(x1 - x0)
    cropped_height = int(y1 - y0)

    min_height = max(1, int(rgb.height * min_fraction))
    min_width = max(1, int(rgb.width * min_fraction))
    if cropped_height < min_height or cropped_width < min_width:
        LOGGER.info(
            "Detected crop %sx%s was too small; keeping original image size %sx%s",
            cropped_width,
            cropped_height,
            rgb.width,
            rgb.height,
        )
        return rgb

    LOGGER.info(
        "Trimmed bright margins from %sx%s to %sx%s",
        rgb.width,
        rgb.height,
        cropped_width,
        cropped_height,
    )
    return rgb.crop((int(x0), int(y0), int(x1), int(y1)))


def extract_pdf_fundus_image(
    pdf_path: Path,
    *,
    crop_margins: bool = True,
) -> Image.Image:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. Install pymupdf in the environment."
        ) from exc

    with fitz.open(str(pdf_path)) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        page = document[0]
        LOGGER.info("Processing %s", pdf_path)
        candidates: list[dict] = []
        for image_entry in page.get_images(full=True):
            xref = image_entry[0]
            info = document.extract_image(xref)
            if info:
                candidates.append(info)

        LOGGER.info("Found %d embedded images in %s", len(candidates), pdf_path.name)
        primary = select_primary_image(candidates)
        image = decode_image_bytes(primary["image"])
        LOGGER.info("Decoded native image size %sx%s", image.width, image.height)
        if crop_margins:
            return crop_bright_margins(image)
        return image


def extract_pdf_directory(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    overwrite: bool = False,
    crop_margins: bool = True,
) -> list[Path]:
    source = Path(source_dir)
    destination = Path(destination_dir)
    pdf_files = iter_pdfs(source)
    if not pdf_files:
        raise ValueError(f"No PDF files found under {source}")

    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting fundus images from %d PDFs in %s", len(pdf_files), source)
    written: list[Path] = []
    for pdf_path in pdf_files:
        output_path = destination / f"{pdf_path.stem}.png"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Destination file already exists: {output_path}")
        image = extract_pdf_fundus_image(pdf_path, crop_margins=crop_margins)
        image.save(output_path)
        LOGGER.info("Wrote %s", output_path)
        written.append(output_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the primary fundus image from first-page PDFs into flat PNG files."
    )
    parser.add_argument("source", type=Path, help="Directory containing PDF files.")
    parser.add_argument(
        "destination",
        type=Path,
        help="Directory where extracted PNG files will be written.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-crop-margins",
        dest="crop_margins",
        action="store_false",
        help="Keep the extracted embedded image exactly as stored without trimming bright margins.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    written = extract_pdf_directory(
        args.source,
        args.destination,
        overwrite=args.overwrite,
        crop_margins=args.crop_margins,
    )
    LOGGER.info("Extracted %d PDF fundus images into %s", len(written), args.destination)
    return 0
