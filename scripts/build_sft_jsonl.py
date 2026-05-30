from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ocr_common import FINAL_DIR, OCR_STAGE2_DIR, first_words, split_sentences, write_jsonl


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "into",
    "have",
    "will",
    "your",
    "about",
    "their",
    "what",
    "when",
    "where",
    "which",
    "also",
    "more",
    "than",
    "then",
    "they",
    "them",
    "were",
    "been",
    "being",
    "are",
    "was",
    "you",
    "how",
    "why",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bootstrap SFT JSONL files from normalized OCR segments.")
    parser.add_argument("--input-dir", type=Path, default=OCR_STAGE2_DIR / "segments", help="Directory containing normalized segment JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=FINAL_DIR, help="Directory for final per-document training exports.")
    parser.add_argument("--doc-id", type=str, default=None, help="Only process the matching document.")
    parser.add_argument("--min-words", type=int, default=80, help="Skip segments that are too short for useful SFT synthesis.")
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


def sentence_tokens(sentence: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", sentence.lower())


def extractive_summary(text: str, *, max_sentences: int = 4, max_words: int = 160) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return first_words(text, max_words)

    token_counts = Counter(
        token
        for sentence in sentences
        for token in sentence_tokens(sentence)
        if len(token) > 2 and token not in STOPWORDS
    )
    scored: list[tuple[int, float, str]] = []
    for index, sentence in enumerate(sentences):
        tokens = sentence_tokens(sentence)
        if not tokens:
            continue
        score = sum(token_counts[token] for token in tokens) / max(len(tokens), 1)
        scored.append((index, score, sentence))
    if not scored:
        return first_words(text, max_words)

    top = sorted(scored, key=lambda item: item[1], reverse=True)[:max_sentences]
    top = sorted(top, key=lambda item: item[0])

    selected: list[str] = []
    current_words = 0
    for _, _, sentence in top:
        sentence_words = len(re.findall(r"\S+", sentence))
        if selected and current_words + sentence_words > max_words:
            break
        selected.append(sentence)
        current_words += sentence_words
    return " ".join(selected).strip() or first_words(text, max_words)


def build_prompt(doc_id: str, section_title: str) -> str:
    if section_title and section_title != doc_id:
        return (
            f"Summarize the section '{section_title}' from '{doc_id}'. "
            "Focus on the main ideas and keep the explanation concise but complete."
        )
    return (
        f"Summarize this excerpt from '{doc_id}'. "
        "Focus on the main ideas and keep the explanation concise but complete."
    )


def make_message(role: str, content: str) -> dict[str, str]:
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tools": "",
        "tool_calls": "",
    }


def build_sft_rows(segment_rows: list[dict], min_words: int) -> list[dict]:
    sft_rows: list[dict] = []
    for row in segment_rows:
        if row.get("word_count", 0) < min_words:
            continue
        assistant_text = extractive_summary(row["text"])
        if not assistant_text:
            continue
        sft_rows.append(
            {
                "conversations": [
                    make_message("user", build_prompt(row["doc_id"], row.get("section_title", row["doc_id"]))),
                    make_message("assistant", assistant_text),
                ]
            }
        )
    return sft_rows


def main() -> None:
    args = parse_args()
    segment_files = iter_segment_files(args.input_dir, args.doc_id)
    if not segment_files:
        raise SystemExit(f"No normalized segment files found in {args.input_dir}")

    for segment_file in segment_files:
        doc_id = segment_file.name.removesuffix(".segments.jsonl")
        rows = build_sft_rows(read_jsonl(segment_file), args.min_words)
        output_dir = args.output_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"sft_{doc_id}_v1.jsonl"
        write_jsonl(output_path, rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
