from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_common import FINAL_DIR, OCR_STAGE2_DIR, ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pretraining JSONL files from normalized OCR segments.")
    parser.add_argument("--input-dir", type=Path, default=OCR_STAGE2_DIR / "segments", help="Directory containing normalized segment JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=FINAL_DIR, help="Directory for final per-document training exports.")
    parser.add_argument("--doc-id", type=str, default=None, help="Only process the matching document.")
    parser.add_argument("--min-words", type=int, default=40, help="Skip very short segments.")
    return parser.parse_args()


def iter_segment_files(input_dir: Path, doc_id: str | None) -> list[Path]:
    if doc_id:
        path = input_dir / f"{doc_id}.segments.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No normalized segments found for doc_id={doc_id!r}")
        return [path]
    return sorted(input_dir.glob("*.segments.jsonl"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pretrain_rows(segment_rows: list[dict], min_words: int) -> list[dict]:
    output_rows: list[dict] = []
    for row in segment_rows:
        if row.get("word_count", 0) < min_words:
            continue
        output_rows.append(
            {
                "text": row["text"],
                "doc_id": row["doc_id"],
                "segment_id": row["segment_id"],
                "source_pdf": row["source_pdf"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "section_title": row["section_title"],
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()
    segment_files = iter_segment_files(args.input_dir, args.doc_id)
    if not segment_files:
        raise SystemExit(f"No normalized segment files found in {args.input_dir}")

    for segment_file in segment_files:
        doc_id = segment_file.name.removesuffix(".segments.jsonl")
        rows = build_pretrain_rows(read_jsonl(segment_file), args.min_words)
        output_dir = ensure_dir(args.output_dir / doc_id)
        output_path = output_dir / f"pretrain_{doc_id}_v1.jsonl"
        write_jsonl(output_path, rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
