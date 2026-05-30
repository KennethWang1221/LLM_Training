from __future__ import annotations

import argparse
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocr_common import OCR_STAGE1_DIR, PDF_DIR, REPO_ROOT, doc_id_from_pdf, ensure_dir, iter_pdf_paths, write_json

try:
    from paddleocr import PaddleOCRVL
except ImportError as exc:  # pragma: no cover - depends on runtime image
    raise SystemExit(
        "paddleocr is not installed. Run this script inside the PaddleOCR container or install PaddleOCR-VL locally."
    ) from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def classify_asset(asset_name: str) -> str:
    lowered = asset_name.lower()
    if "table" in lowered:
        return "table"
    if "formula" in lowered or "equation" in lowered:
        return "formula"
    return "img"


def save_page_assets(
    *,
    doc_id: str,
    page_index: int,
    page_markdown: str,
    markdown_images: dict[str, Any],
    assets_dir: Path,
) -> tuple[str, list[str]]:
    saved_paths: list[str] = []
    counters: dict[str, int] = {"img": 0, "table": 0, "formula": 0}
    rewritten = page_markdown

    for original_path, image in sorted(markdown_images.items()):
        kind = classify_asset(original_path)
        counters[kind] += 1
        suffix = Path(original_path).suffix or ".png"
        asset_name = f"{doc_id}.p{page_index:04d}.{kind}{counters[kind]:03d}{suffix}"
        asset_path = assets_dir / asset_name
        image.save(asset_path)
        rewritten = rewritten.replace(original_path, f"./{doc_id}.assets/{asset_name}")
        saved_paths.append(relative_to_repo(asset_path))

    return rewritten, saved_paths


def write_page_outputs(
    *,
    doc_id: str,
    pdf_path: Path,
    json_dir: Path,
    markdown_pages_dir: Path,
    assets_dir: Path,
    page_index: int,
    page_result: Any,
) -> dict[str, Any]:
    page_json_path = json_dir / f"{doc_id}.p{page_index:04d}.json"
    page_markdown_path = markdown_pages_dir / f"{doc_id}.p{page_index:04d}.md"

    page_json = getattr(page_result, "json", {})
    write_json(page_json_path, page_json)

    markdown_info = dict(getattr(page_result, "markdown", {}) or {})
    markdown_text = markdown_info.get("markdown_texts", "") or ""
    markdown_images = markdown_info.get("markdown_images", {}) or {}
    rewritten_markdown, asset_paths = save_page_assets(
        doc_id=doc_id,
        page_index=page_index,
        page_markdown=markdown_text,
        markdown_images=markdown_images,
        assets_dir=assets_dir,
    )
    markdown_info["markdown_texts"] = rewritten_markdown
    markdown_info["markdown_images"] = {}

    page_header = "\n".join(
        [
            f"<!-- DOC_ID: {doc_id} -->",
            f"<!-- SOURCE_PDF: {relative_to_repo(pdf_path)} -->",
            f"<!-- PAGE: {page_index:04d} -->",
            "",
        ]
    )
    page_markdown_path.write_text(page_header + rewritten_markdown.strip() + "\n", encoding="utf-8")

    return {
        "page_index": page_index,
        "json_path": relative_to_repo(page_json_path),
        "markdown_path": relative_to_repo(page_markdown_path),
        "assets": asset_paths,
        "markdown_info": markdown_info,
    }


def process_pdf(pdf_path: Path, pipeline: PaddleOCRVL, output_dir: Path, overwrite: bool) -> Path:
    doc_id = doc_id_from_pdf(pdf_path)
    doc_dir = output_dir / doc_id
    if doc_dir.exists():
        if not overwrite:
            manifest_path = doc_dir / f"{doc_id}.manifest.json"
            raise FileExistsError(f"{manifest_path} already exists. Pass --overwrite to rebuild {doc_id}.")
        shutil.rmtree(doc_dir)
    doc_dir = ensure_dir(doc_dir)
    json_dir = ensure_dir(doc_dir / f"{doc_id}.json")
    markdown_pages_dir = ensure_dir(doc_dir / f"{doc_id}.markdown_pages")
    assets_dir = ensure_dir(doc_dir / f"{doc_id}.assets")
    manifest_path = doc_dir / f"{doc_id}.manifest.json"
    merged_markdown_path = doc_dir / f"{doc_id}.markdown_merged.md"

    manifest: dict[str, Any] = {
        "doc_id": doc_id,
        "source_pdf": relative_to_repo(pdf_path),
        "status": "started",
        "started_at": utc_now(),
        "outputs": {
            "doc_dir": relative_to_repo(doc_dir),
            "json_dir": relative_to_repo(json_dir),
            "markdown_pages_dir": relative_to_repo(markdown_pages_dir),
            "assets_dir": relative_to_repo(assets_dir),
            "merged_markdown_path": relative_to_repo(merged_markdown_path),
        },
    }
    write_json(manifest_path, manifest)

    try:
        results = pipeline.predict(input=str(pdf_path))
        page_records = []
        markdown_infos = []
        asset_count = 0

        for page_index, page_result in enumerate(results, start=1):
            page_record = write_page_outputs(
                doc_id=doc_id,
                pdf_path=pdf_path,
                json_dir=json_dir,
                markdown_pages_dir=markdown_pages_dir,
                assets_dir=assets_dir,
                page_index=page_index,
                page_result=page_result,
            )
            markdown_infos.append(page_record.pop("markdown_info"))
            asset_count += len(page_record["assets"])
            page_records.append(page_record)

        merged_markdown = pipeline.concatenate_markdown_pages(markdown_infos)
        merged_header = "\n".join(
            [
                f"<!-- DOC_ID: {doc_id} -->",
                f"<!-- SOURCE_PDF: {relative_to_repo(pdf_path)} -->",
                "",
            ]
        )
        merged_markdown_path.write_text(merged_header + merged_markdown.strip() + "\n", encoding="utf-8")

        manifest.update(
            {
                "status": "success",
                "completed_at": utc_now(),
                "page_count": len(page_records),
                "asset_count": asset_count,
                "pages": page_records,
            }
        )
    except Exception as exc:  # pragma: no cover - depends on runtime OCR
        manifest.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PaddleOCR-VL artifacts from PDFs.")
    parser.add_argument("--input-dir", type=Path, default=PDF_DIR, help="Directory containing source PDFs.")
    parser.add_argument("--output-dir", type=Path, default=OCR_STAGE1_DIR, help="Directory for OCR stage1 outputs.")
    parser.add_argument("--doc-id", type=str, default=None, help="Only process the PDF whose stem matches this value.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing stage1 outputs for the selected PDFs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    pdf_paths = iter_pdf_paths(args.input_dir, args.doc_id)
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {args.input_dir}")

    pipeline = PaddleOCRVL()
    for pdf_path in pdf_paths:
        manifest_path = process_pdf(pdf_path=pdf_path, pipeline=pipeline, output_dir=args.output_dir, overwrite=args.overwrite)
        print(f"Wrote {relative_to_repo(manifest_path)}")


if __name__ == "__main__":
    main()
