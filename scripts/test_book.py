import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.orchestrator import DocumentPipelineOrchestrator, calculate_file_sha256
from app.schemas.document import DocumentStatus


def render_progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return f"[{' ' * width}] 0%"
    fraction = min(1.0, current / total)
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    percent = int(round(fraction * 100))
    return f"[{bar}] {percent}%"


async def run_single_book_test(
    pdf_path: str,
    output_dir: str | None = None,
    document_id: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    debug: bool = False,
    force: bool = False,
    generate_toon: bool = False,
    toon_output: str | None = None,
) -> int:
    p_path = Path(pdf_path)
    if not p_path.exists():
        print(f"Error: PDF file '{pdf_path}' does not exist.", file=sys.stderr)
        return 1

    file_sha256 = calculate_file_sha256(p_path)
    doc_id = document_id or f"doc_{file_sha256[:16]}"
    out_dir = Path(output_dir) if output_dir else Path("./data/processed") / doc_id

    print("=" * 50)
    print(" Textbook OCR - Single Book Test")
    print("=" * 50)
    print(f"\nInput:\n  {p_path.name} (ID: {doc_id})")

    orchestrator = DocumentPipelineOrchestrator()

    # Step 1: Profiling
    print("\nAnalyzing PDF...")
    t0_prof = time.perf_counter()
    profile = orchestrator.analyzer.profile_document(p_path)
    t_prof = time.perf_counter() - t0_prof
    print(f"  Pages: {profile.page_count}")
    print(f"  Text-layer pages: {profile.text_layer_pages}")
    print(f"  Poor-text pages: {profile.poor_text_pages}")
    print(f"  Image / Scanned pages: {profile.zero_text_pages + profile.image_pages}")
    print(f"  Profiling completed in {t_prof:.2f}s")

    # Step 2: Processing with progress bar
    print("\nProcessing pages...")

    def progress_callback(progress: float, current: int, total: int):
        bar = render_progress_bar(current, total)
        sys.stdout.write(f"\r  {bar} ({current}/{total} pages)")
        sys.stdout.flush()

    manifest = await orchestrator.process_document(
        pdf_path=p_path,
        output_dir=out_dir,
        document_id=doc_id,
        start_page=start_page,
        end_page=end_page,
        debug=debug,
        force=force,
        generate_toon=generate_toon,
        toon_output_path=toon_output,
        progress_callback=progress_callback,
    )
    print("\n")

    # Step 3: Load report for summary
    report_file = out_dir / "report.json"
    report = {}
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as rf:
            report = json.load(rf)

    extraction = report.get("extraction", {})
    structure = report.get("structure", {})
    proc_info = report.get("processing", {})

    print("Extraction:")
    print(f"  pdfplumber: {extraction.get('pdfplumber', 0)}")
    print(f"  baseline OCR: {extraction.get('baseline_ocr', 0)}")
    print(f"  CNN OCR: {extraction.get('cnn_ocr', 0)}")
    print(f"  failed: {manifest.failed_pages}")

    print("\nStructure:")
    print(f"  Chapters: {structure.get('chapters', manifest.chapters)}")
    print(f"  Sections: {structure.get('sections', manifest.sections)}")
    print(f"  Chunks: {structure.get('chunks', manifest.chunks)}")

    audit_dir = out_dir / "audit"
    if audit_dir.exists():
        print(f"\nAudit Report:\n  {audit_dir / 'structural_validation.md'}")

    if generate_toon:
        expected_toon = Path(toon_output) if toon_output else out_dir / f"{p_path.stem}_parsed.toon"
        print(f"\nTOON Output:\n  {expected_toon}")

    print(f"\nOutput:\n  {out_dir}")
    if debug:
        print(f"  Debug Visualizations: {out_dir / 'debug'}")

    final_status = proc_info.get("status", "COMPLETED").upper()
    print(f"\nStatus:\n  {final_status}")
    print("=" * 50)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Textbook OCR Single-Book Benchmark Runner (Document-Agnostic)"
    )
    parser.add_argument("pdf_path", type=str, nargs="?", help="Path to PDF textbook file")
    parser.add_argument("--input", "-i", type=str, dest="input_pdf", help="Alternative PDF path flag")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom output directory")
    parser.add_argument("--document-id", type=str, default=None, help="Custom document ID override")
    parser.add_argument("--start-page", type=int, default=None, help="1-indexed start page")
    parser.add_argument("--end-page", type=int, default=None, help="1-indexed end page")
    parser.add_argument("--debug", action="store_true", help="Generate debug visualization images")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of existing pages")
    parser.add_argument("--toon", action="store_true", help="Generate single canonical .toon representation")
    parser.add_argument("--toon-output", type=str, default=None, help="Custom .toon output file path")

    args = parser.parse_args()
    target_path = args.pdf_path or args.input_pdf
    if not target_path:
        parser.print_help()
        sys.exit(1)

    exit_code = asyncio.run(
        run_single_book_test(
            pdf_path=target_path,
            output_dir=args.output,
            document_id=args.document_id,
            start_page=args.start_page,
            end_page=args.end_page,
            debug=args.debug,
            force=args.force,
            generate_toon=args.toon,
            toon_output=args.toon_output,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
