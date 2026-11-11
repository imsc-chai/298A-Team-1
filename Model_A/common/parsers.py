"""Local-file parsers: PDF, PNG, Markdown, CSV -> text chunks.

The same parser output feeds both the retriever index and the read_local_file tool, so every
model sees identical file contents (fair comparison) and evidence ids line up with chunks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps
from pypdf import PdfReader

from common.config import cache_dir

# pypdf warns about benign xref quirks in the synthetic PDFs; the extracted text is correct.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Upscale + autocontrast + block layout: on Maya's flight_ticket.png this recovers airline,
# flight number, times, date, seat and confirmation code that raw OCR garbles.
OCR_SCALE = 2
OCR_CONFIG = "--oem 1 --psm 6"
PARSER_VERSION = "1"  # bump to invalidate the parse cache when parsing logic changes

MAX_CHUNK_CHARS = 1200


class UnsupportedFileType(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    index: int  # 1-based, used in evidence ids (file_<stem>_001)
    heading: str
    text: str


@dataclass(frozen=True)
class ParsedFile:
    name: str
    kind: str
    text: str  # full text, what read_local_file returns
    chunks: list[Chunk]


def file_record_id(file_name: str, chunk_index: int) -> str:
    stem = Path(file_name).stem
    return f"file_{stem}_{chunk_index:03d}"


def parse_file(path: Path, use_cache: bool = True) -> ParsedFile:
    """Parse one local file, caching by content hash because OCR is the slow step."""
    path = Path(path)
    raw = path.read_bytes()
    key = hashlib.sha256(raw + PARSER_VERSION.encode() + path.name.encode()).hexdigest()
    cache_file = cache_dir() / "parsed" / f"{key}.json"
    if use_cache and cache_file.is_file():
        return _from_json(cache_file.read_text())

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _pdf_text(raw)
        chunks = _paragraph_chunks(text, heading=path.name)
    elif suffix == ".png":
        text = _png_text(path)
        chunks = _paragraph_chunks(text, heading=path.name)
    elif suffix == ".md":
        text = raw.decode("utf-8")
        chunks = _markdown_chunks(text, default_heading=path.name)
    elif suffix == ".csv":
        text = raw.decode("utf-8")
        chunks = _csv_chunks(text, file_name=path.name)
    else:
        raise UnsupportedFileType(f"No parser for {path.name}")

    parsed = ParsedFile(name=path.name, kind=suffix.lstrip("."), text=text.strip(), chunks=chunks)
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(asdict(parsed)))
    return parsed


def _from_json(blob: str) -> ParsedFile:
    data = json.loads(blob)
    return ParsedFile(
        name=data["name"], kind=data["kind"], text=data["text"], chunks=[Chunk(**c) for c in data["chunks"]]
    )


def _pdf_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _png_text(path: Path) -> str:
    image = ImageOps.grayscale(Image.open(path).convert("RGB"))
    image = image.resize((image.width * OCR_SCALE, image.height * OCR_SCALE), Image.LANCZOS)
    image = ImageOps.autocontrast(image)
    return pytesseract.image_to_string(image, config=OCR_CONFIG)


def _paragraph_chunks(text: str, heading: str) -> list[Chunk]:
    """Pack lines into ~MAX_CHUNK_CHARS chunks; these documents are 1-2 pages each."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) > MAX_CHUNK_CHARS:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return [Chunk(index=i + 1, heading=heading, text=c) for i, c in enumerate(chunks)]


def _markdown_chunks(text: str, default_heading: str) -> list[Chunk]:
    """One chunk per heading section (oversized sections are split further)."""
    sections: list[tuple[str, list[str]]] = []
    heading = default_heading
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.*)", line)
        if match:
            if any(b.strip() for b in body):
                sections.append((heading, body))
            heading, body = match.group(1).strip(), [line]
        else:
            body.append(line)
    if any(b.strip() for b in body):
        sections.append((heading, body))

    chunks: list[Chunk] = []
    for sec_heading, sec_lines in sections:
        for piece in _paragraph_chunks("\n".join(sec_lines), sec_heading):
            chunks.append(Chunk(index=len(chunks) + 1, heading=sec_heading, text=piece.text))
    return chunks


def _csv_chunks(text: str, file_name: str) -> list[Chunk]:
    """One chunk per row, rendered as header=value pairs so a row is self-describing."""
    rows = list(csv.DictReader(io.StringIO(text)))
    chunks = []
    for i, row in enumerate(rows):
        rendered = "; ".join(f"{k}={v}" for k, v in row.items())
        chunks.append(Chunk(index=i + 1, heading=f"{file_name} row {i + 1}", text=rendered))
    return chunks
