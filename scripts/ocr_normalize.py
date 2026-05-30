from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ocr_common import (
    OCR_STAGE1_DIR,
    OCR_STAGE2_DIR,
    PDF_DIR,
    extract_page_number,
    first_words,
    normalize_text,
    read_json,
    word_count,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize OCR stage1 artifacts into clean documents and segments.")
    parser.add_argument("--input-dir", type=Path, default=OCR_STAGE1_DIR, help="Directory containing stage1 OCR outputs.")
    parser.add_argument("--output-dir", type=Path, default=OCR_STAGE2_DIR, help="Directory for normalized artifacts.")
    parser.add_argument("--doc-id", type=str, default=None, help="Only process the matching document directory.")
    parser.add_argument("--target-words", type=int, default=900, help="Target word count per segment.")
    parser.add_argument("--max-words", type=int, default=1400, help="Hard cap per segment before forcing a new chunk.")
    return parser.parse_args()


def iter_doc_dirs(input_dir: Path, doc_id: str | None) -> list[Path]:
    if doc_id:
        path = input_dir / doc_id
        if not path.exists():
            raise FileNotFoundError(f"No OCR stage1 directory found for doc_id={doc_id!r}")
        return [path]
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def detect_doc_id(doc_dir: Path) -> str:
    return doc_dir.name


def is_special_markdown_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return stripped.startswith(("#", "-", "*", "|", ">", "```"))


def join_wrapped_lines(text: str) -> str:
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        if any(is_special_markdown_line(line) for line in current):
            blocks.append("\n".join(current).strip())
        else:
            joined = " ".join(line.strip() for line in current if line.strip())
            blocks.append(joined.strip())
        current.clear()

    for line in lines:
        if not line.strip():
            flush()
            blocks.append("")
        else:
            current.append(line)
    flush()

    rebuilt: list[str] = []
    for block in blocks:
        if not block:
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
        else:
            rebuilt.append(block)
    return "\n".join(rebuilt).strip()


def clean_markdown(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned)
    cleaned = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = join_wrapped_lines(cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def detect_repeated_edge_lines(pages: list[dict[str, Any]], *, edge: str) -> set[str]:
    values: list[str] = []
    for page in pages:
        lines = nonempty_lines(page["raw_text"])
        if not lines:
            continue
        value = lines[0] if edge == "top" else lines[-1]
        if 0 < len(value) <= 120:
            values.append(value)
    if not values:
        return set()

    threshold = max(2, math.ceil(len(pages) * 0.6))
    counts = Counter(values)
    return {value for value, count in counts.items() if count >= threshold}


def strip_repeated_edge_lines(text: str, *, repeated_top: set[str], repeated_bottom: set[str]) -> str:
    lines = [line for line in text.splitlines()]
    while lines and lines[0].strip() in repeated_top:
        lines.pop(0)
    while lines and lines[-1].strip() in repeated_bottom:
        lines.pop()
    return "\n".join(lines).strip()


def load_pages(doc_dir: Path, doc_id: str) -> list[dict[str, Any]]:
    pages_dir = doc_dir / f"{doc_id}.markdown_pages"
    page_paths = sorted(pages_dir.glob("*.md"))
    pages: list[dict[str, Any]] = []
    for path in page_paths:
        raw_text = path.read_text(encoding="utf-8")
        pages.append(
            {
                "page_index": extract_page_number(path),
                "path": path,
                "raw_text": raw_text,
            }
        )
    return pages


def paragraphs_from_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paragraph_records: list[dict[str, Any]] = []
    for page in pages:
        for paragraph in [part.strip() for part in page["clean_text"].split("\n\n") if part.strip()]:
            paragraph_records.append(
                {
                    "page_index": page["page_index"],
                    "text": paragraph,
                }
            )
    return paragraph_records


def split_into_sections(paragraphs: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title = doc_id
    current_entries: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_entries, current_title
        if not current_entries:
            return
        text = normalize_text("\n\n".join(entry["text"] for entry in current_entries if entry["text"].strip()))
        if text:
            sections.append(
                {
                    "section_title": current_title,
                    "page_start": current_entries[0]["page_index"],
                    "page_end": current_entries[-1]["page_index"],
                    "paragraphs": list(current_entries),
                    "text": text,
                }
            )
        current_entries = []

    for paragraph in paragraphs:
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", paragraph["text"].strip())
        if heading_match:
            flush()
            current_title = heading_match.group(2).strip()
            current_entries = [paragraph]
        else:
            current_entries.append(paragraph)

    flush()

    return sections or [
        {
            "section_title": doc_id,
            "page_start": paragraphs[0]["page_index"] if paragraphs else 1,
            "page_end": paragraphs[-1]["page_index"] if paragraphs else 1,
            "paragraphs": list(paragraphs),
            "text": normalize_text("\n\n".join(paragraph["text"] for paragraph in paragraphs)),
        }
    ]


def chunk_section(
    section: dict[str, Any],
    *,
    target_words: int,
    max_words: int,
    doc_id: str,
    source_pdf: str,
    start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    paragraphs = list(section["paragraphs"])
    segments: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    segment_index = start_index

    def flush() -> None:
        nonlocal buffer, segment_index
        if not buffer:
            return
        text = "\n\n".join(paragraph["text"] for paragraph in buffer).strip()
        if text:
            segments.append(
                {
                    "doc_id": doc_id,
                    "segment_id": f"{doc_id}.seg{segment_index:04d}",
                    "source_pdf": source_pdf,
                    "page_start": buffer[0]["page_index"],
                    "page_end": buffer[-1]["page_index"],
                    "section_title": section["section_title"],
                    "text": text,
                    "word_count": word_count(text),
                    "preview": first_words(text, 32),
                }
            )
            segment_index += 1
        buffer = []

    for paragraph in paragraphs:
        candidate = "\n\n".join(entry["text"] for entry in (buffer + [paragraph])).strip()
        candidate_words = word_count(candidate)
        if buffer and candidate_words > max_words:
            flush()
            buffer = [paragraph]
            continue
        buffer.append(paragraph)
        if word_count("\n\n".join(entry["text"] for entry in buffer)) >= target_words:
            flush()
    flush()
    return segments, segment_index


def normalize_doc(doc_dir: Path, output_dir: Path, *, target_words: int, max_words: int) -> dict[str, Any]:
    doc_id = detect_doc_id(doc_dir)
    manifest_path = doc_dir / f"{doc_id}.manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    pages = load_pages(doc_dir, doc_id)
    if not pages:
        raise FileNotFoundError(f"No page markdown files found under {doc_dir}")

    repeated_top = detect_repeated_edge_lines(pages, edge="top")
    repeated_bottom = detect_repeated_edge_lines(pages, edge="bottom")
    cleaned_pages: list[dict[str, Any]] = []
    empty_pages: list[int] = []

    for page in pages:
        trimmed = strip_repeated_edge_lines(page["raw_text"], repeated_top=repeated_top, repeated_bottom=repeated_bottom)
        clean_text = clean_markdown(trimmed)
        page_record = {
            "page_index": page["page_index"],
            "clean_text": clean_text,
            "word_count": word_count(clean_text),
        }
        if not clean_text:
            empty_pages.append(page["page_index"])
        cleaned_pages.append(page_record)

    source_pdf = manifest.get("source_pdf", str((PDF_DIR / f"{doc_id}.pdf").relative_to(PDF_DIR.parent.parent)))
    paragraphs = paragraphs_from_pages(cleaned_pages)
    sections = split_into_sections(paragraphs, doc_id)
    segments: list[dict[str, Any]] = []
    segment_index = 1
    for section in sections:
        section_segments, segment_index = chunk_section(
            section,
            target_words=target_words,
            max_words=max_words,
            doc_id=doc_id,
            source_pdf=source_pdf,
            start_index=segment_index,
        )
        segments.extend(section_segments)

    documents_dir = output_dir / "documents"
    segments_dir = output_dir / "segments"
    reports_dir = output_dir / "reports"
    documents_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    document_md_path = documents_dir / f"{doc_id}.document.md"
    document_json_path = documents_dir / f"{doc_id}.document.json"
    segments_path = segments_dir / f"{doc_id}.segments.jsonl"
    qc_path = reports_dir / f"{doc_id}.qc.json"
    stats_path = reports_dir / f"{doc_id}.stats.json"

    document_markdown_parts: list[str] = []
    for page in cleaned_pages:
        document_markdown_parts.append(f"<!-- PAGE: {page['page_index']:04d} -->")
        document_markdown_parts.append(page["clean_text"])
        document_markdown_parts.append("")
    document_md_path.write_text("\n".join(document_markdown_parts).strip() + "\n", encoding="utf-8")

    segment_count = write_jsonl(segments_path, segments)
    document_payload = {
        "doc_id": doc_id,
        "source_pdf": source_pdf,
        "page_count": len(cleaned_pages),
        "segment_count": segment_count,
        "document_markdown_path": str(document_md_path.relative_to(output_dir.parent.parent)),
        "segments_path": str(segments_path.relative_to(output_dir.parent.parent)),
    }
    write_json(document_json_path, document_payload)

    write_json(
        qc_path,
        {
            "doc_id": doc_id,
            "empty_pages": empty_pages,
            "repeated_top_lines_removed": sorted(repeated_top),
            "repeated_bottom_lines_removed": sorted(repeated_bottom),
            "short_segments": [segment["segment_id"] for segment in segments if segment["word_count"] < 80],
        },
    )
    write_json(
        stats_path,
        {
            "doc_id": doc_id,
            "page_count": len(cleaned_pages),
            "segment_count": segment_count,
            "total_words": sum(page["word_count"] for page in cleaned_pages),
            "avg_words_per_page": round(sum(page["word_count"] for page in cleaned_pages) / max(len(cleaned_pages), 1), 2),
            "avg_words_per_segment": round(sum(segment["word_count"] for segment in segments) / max(segment_count, 1), 2),
        },
    )

    return {
        "doc_id": doc_id,
        "document_markdown_path": document_md_path,
        "document_json_path": document_json_path,
        "segments_path": segments_path,
        "qc_path": qc_path,
        "stats_path": stats_path,
    }


def main() -> None:
    args = parse_args()
    doc_dirs = iter_doc_dirs(args.input_dir, args.doc_id)
    if not doc_dirs:
        raise SystemExit(f"No stage1 directories found in {args.input_dir}")

    for doc_dir in doc_dirs:
        outputs = normalize_doc(doc_dir, args.output_dir, target_words=args.target_words, max_words=args.max_words)
        print(f"Wrote {outputs['segments_path']}")


if __name__ == "__main__":
    main()
