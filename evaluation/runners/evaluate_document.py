import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import (
    ChunkEvaluator,
    LayoutEvaluator,
    OCREvaluator,
    ReadingOrderEvaluator,
    StructureEvaluator,
    ToonEvaluator,
)
from evaluation.reports.report_generator import EvaluationReportGenerator
from evaluation.schemas.ground_truth import DocumentGT


def discover_processed_document(document_dir: Path) -> dict[str, Any]:
    if not document_dir.exists() or not document_dir.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {document_dir}")

    doc_id = document_dir.name
    filename = "unknown"
    manifest_data = {}
    metadata_data = {}
    report_data = {}

    manifest_path = document_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            doc_id = manifest_data.get("document_id", doc_id)
            filename = manifest_data.get("filename", filename)

    metadata_path = document_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_data = json.load(f)
            doc_id = metadata_data.get("document_id", doc_id)
            filename = metadata_data.get("filename", filename)

    report_path = document_dir / "report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

    # 1. Load pages
    pages = []
    pages_dir = document_dir / "pages"
    if pages_dir.exists():
        for pf in sorted(pages_dir.glob("*.json")):
            with open(pf, "r", encoding="utf-8") as f:
                pages.append(json.load(f))

    # 2. Load chapters
    chapters = []
    chapters_dir = document_dir / "chapters"
    if chapters_dir.exists():
        for cf in sorted(chapters_dir.glob("*.json")):
            with open(cf, "r", encoding="utf-8") as f:
                chapters.append(json.load(f))

    # 3. Load chunks
    chunks = []
    chunks_dir = document_dir / "chunks"
    if chunks_dir.exists():
        for chkf in sorted(chunks_dir.glob("*.jsonl")):
            with open(chkf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line.strip()))

    # 4. Discover TOON file
    toon_files = list(document_dir.glob("*.toon"))
    toon_path = toon_files[0] if toon_files else None

    return {
        "document_id": doc_id,
        "filename": filename,
        "document_dir": str(document_dir),
        "manifest": manifest_data,
        "metadata": metadata_data,
        "report": report_data,
        "pages": pages,
        "chapters": chapters,
        "chunks": chunks,
        "toon_path": str(toon_path) if toon_path else None,
    }


def load_ground_truth(gt_path_str: Optional[str]) -> Optional[DocumentGT]:
    if not gt_path_str:
        return None

    p = Path(gt_path_str)
    if not p.exists():
        print(f"Warning: Ground truth path not found: {p}", file=sys.stderr)
        return None

    target_file = p
    if p.is_dir():
        candidates = list(p.glob("*.json"))
        if candidates:
            target_file = candidates[0]
        else:
            print(f"Warning: No JSON ground truth files found in directory: {p}", file=sys.stderr)
            return None

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return DocumentGT.model_validate(data)
    except Exception as e:
        print(f"Warning: Failed to parse ground truth at {target_file}: {e}", file=sys.stderr)
        return None


def run_evaluation(
    document_path: str | Path,
    ground_truth_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    generate_html: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    doc_path = Path(document_path)
    doc_data = discover_processed_document(doc_path)
    gt = load_ground_truth(str(ground_truth_path) if ground_truth_path else None)

    doc_id = doc_data["document_id"]
    filename = doc_data["filename"]
    pages = doc_data["pages"]
    chapters = doc_data["chapters"]
    chunks = doc_data["chunks"]
    toon_path = doc_data["toon_path"]
    manifest = doc_data["manifest"]

    gt_available = gt is not None
    eval_mode = "GROUND_TRUTH_EVALUATION" if gt_available else "SYSTEM_VALIDATION"

    # Instantiate evaluators
    ocr_eval = OCREvaluator()
    layout_eval = LayoutEvaluator()
    struct_eval = StructureEvaluator()
    order_eval = ReadingOrderEvaluator()
    chunk_eval = ChunkEvaluator()
    toon_eval = ToonEvaluator()

    # Execute evaluations
    ocr_res = ocr_eval.evaluate(pages, gt)
    layout_res = layout_eval.evaluate(pages, gt)
    struct_res = struct_eval.evaluate(chapters, pages, chunks, gt)
    order_res = order_eval.evaluate(pages, gt)
    chunk_res = chunk_eval.evaluate(chunks, chapters, max_page=len(pages), ground_truth=gt)
    toon_res = toon_eval.evaluate(
        toon_path=toon_path,
        expected_document_id=doc_id,
        manifest_data=manifest,
        chapters=chapters,
        chunks=chunks,
        pages=pages,
    )

    evaluation_payload = {
        "document_id": doc_id,
        "filename": filename,
        "evaluation_mode": eval_mode,
        "ground_truth_available": gt_available,
        "processing_statistics": {
            "total_pages": len(pages),
            "total_chapters": len(chapters),
            "total_sections": sum(len(ch.get("sections", [])) for ch in chapters),
            "total_chunks": len(chunks),
        },
        "output_integrity": {
            "has_manifest": bool(manifest),
            "has_pages": len(pages) > 0,
            "has_chapters": len(chapters) > 0,
            "has_chunks": len(chunks) > 0,
            "has_toon": toon_res.get("toon_file_found", False),
        },
        "ocr_metrics": ocr_res,
        "layout_metrics": layout_res,
        "structure_metrics": struct_res,
        "reading_order_metrics": order_res,
        "chunk_metrics": chunk_res,
        "toon_metrics": toon_res,
        "warnings": [],
        "errors": [],
    }

    # Resolve output directory
    target_out = (
        Path(output_dir)
        if output_dir
        else Path("./data/evaluation_results") / doc_id
    )
    target_out.mkdir(parents=True, exist_ok=True)

    report_gen = EvaluationReportGenerator()
    report_gen.generate_json_reports(target_out, evaluation_payload)

    html_out = None
    if generate_html:
        html_out = report_gen.generate_html_report(target_out, evaluation_payload)

    # Print Summary Table
    print("=" * 60)
    print(" OCR & Document Pipeline Evaluation")
    print("=" * 60)
    print(f"Document ID:             {doc_id}")
    print(f"Filename:                {filename}")
    print(f"Mode:                    {eval_mode}")
    print(f"Ground Truth:            {'Available' if gt_available else 'Not Supplied'}")
    print("-" * 60)
    print("Summary Metrics:")
    if gt_available:
        print(f"  OCR CER:               {ocr_res.get('cer', 'N/A')}")
        print(f"  OCR WER:               {ocr_res.get('wer', 'N/A')}")
        print(f"  Layout F1:             {layout_res.get('overall_f1', 'N/A')}")
        print(f"  Tree Similarity:       {struct_res.get('normalized_tree_similarity', 'N/A')}")
        print(f"  Pairwise Order Acc:    {order_res.get('pairwise_ordering_accuracy', 'N/A')}")
        print(f"  Chunk Boundary F1:     {chunk_res.get('boundary_f1', 'N/A')}")
    else:
        print(f"  Parent-Child Score:    {struct_res.get('parent_child_integrity', 1.0)}")
        print(f"  Boundary Integrity:    {struct_res.get('page_boundary_integrity', 1.0)}")
        print(f"  Reading Order Score:   {order_res.get('reading_order_consistency_score', 1.0)}")
        print(f"  Chunk Integrity Score: {chunk_res.get('chunk_integrity_score', 1.0)}")
        print(f"  TOON Valid:            {toon_res.get('valid', False)}")

    print("-" * 60)
    print(f"Results Directory:       {target_out}")
    if html_out:
        print(f"HTML Report:             {html_out}")
    print("=" * 60)

    return evaluation_payload


def main():
    parser = argparse.ArgumentParser(
        description="Textbook OCR Document Evaluation Harness (Document-Agnostic)"
    )
    parser.add_argument("document_path", type=str, help="Path to processed document directory, e.g. data/processed/<document_id>")
    parser.add_argument("--ground-truth", "-g", type=str, default=None, help="Path to ground truth JSON file or directory")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom output directory for evaluation results")
    parser.add_argument("--html", action="store_true", default=True, help="Generate HTML report (default: True)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose evaluation logging")

    args = parser.parse_args()
    run_evaluation(
        document_path=args.document_path,
        ground_truth_path=args.ground_truth,
        output_dir=args.output,
        generate_html=args.html,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
