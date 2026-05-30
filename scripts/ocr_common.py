from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
PDF_DIR = DATASET_DIR / "pdfs"
OCR_STAGE1_DIR = DATASET_DIR / "ocr_stage1"
OCR_STAGE2_DIR = DATASET_DIR / "ocr_stage2"
FINAL_DIR = DATASET_DIR / "final"

PAGE_FILE_RE = re.compile(r"\.p(?P<page>\d{4})\.")


def doc_id_from_pdf(pdf_path: Path) -> str:
    return pdf_path.stem


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_pdf_paths(input_dir: Path, doc_id: str | None = None) -> list[Path]:
    if doc_id:
        matches = sorted(path for path in input_dir.glob("*.pdf") if path.stem == doc_id)
        if not matches:
            raise FileNotFoundError(f"No PDF found for doc_id={doc_id!r} in {input_dir}")
        return matches
    return sorted(input_dir.glob("*.pdf"))


def extract_page_number(path: Path) -> int:
    match = PAGE_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not extract page number from {path.name}")
    return int(match.group("page"))


def strip_control_chars(text: str) -> str:
    return "".join(
        char
        for char in text
        if char in ("\n", "\t") or unicodedata.category(char)[0] != "C"
    )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = strip_control_chars(normalized)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def first_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).strip() + " ..."
