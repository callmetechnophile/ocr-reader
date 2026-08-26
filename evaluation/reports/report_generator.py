import json
from pathlib import Path
from typing import Any


class EvaluationReportGenerator:
    """
    Serializes evaluation results to JSON files and generates an HTML report.
    """

    def generate_json_reports(self, output_dir: Path, evaluation_data: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Main summary evaluation.json
        with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

        # 2. Individual metric JSON files
        metric_files = {
            "ocr_metrics.json": evaluation_data.get("ocr_metrics", {}),
            "layout_metrics.json": evaluation_data.get("layout_metrics", {}),
            "structure_metrics.json": evaluation_data.get("structure_metrics", {}),
            "reading_order_metrics.json": evaluation_data.get("reading_order_metrics", {}),
            "chunk_metrics.json": evaluation_data.get("chunk_metrics", {}),
            "toon_metrics.json": evaluation_data.get("toon_metrics", {}),
        }

        for fname, data in metric_files.items():
            with open(output_dir / fname, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def generate_html_report(self, output_dir: Path, evaluation_data: dict[str, Any]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        html_file = output_dir / "report.html"

        doc_id = evaluation_data.get("document_id", "unknown")
        filename = evaluation_data.get("filename", "unknown")
        mode = evaluation_data.get("evaluation_mode", "SYSTEM_VALIDATION")
        gt_available = evaluation_data.get("ground_truth_available", False)

        ocr = evaluation_data.get("ocr_metrics", {})
        layout = evaluation_data.get("layout_metrics", {})
        struct = evaluation_data.get("structure_metrics", {})
        order = evaluation_data.get("reading_order_metrics", {})
        chunks = evaluation_data.get("chunk_metrics", {})
        toon = evaluation_data.get("toon_metrics", {})

        def val_display(val: Any) -> str:
            if val is None:
                return "<span class='text-muted'>N/A</span>"
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OCR Evaluation Report — {doc_id}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --success: #4ade80;
      --warning: #facc15;
      --danger: #f87171;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }}
    h1 {{
      margin: 0 0 0.5rem 0;
      font-size: 1.875rem;
      color: var(--accent);
    }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.875rem;
      font-weight: 600;
      margin-right: 0.5rem;
    }}
    .badge-mode {{ background: #0369a1; color: #e0f2fe; }}
    .badge-gt {{ background: {'#166534' if gt_available else '#854d0e'}; color: {'#dcfce7' if gt_available else '#fef9c3'}; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 1.25rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.5rem;
    }}
    td, th {{
      padding: 0.5rem 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 0.9375rem;
    }}
    td:last-child {{
      text-align: right;
      font-weight: 500;
    }}
    .text-muted {{ color: var(--text-muted); }}
    .footer {{
      margin-top: 3rem;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.875rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Document OCR Evaluation Report</h1>
      <div>
        <span class="badge badge-mode">{mode}</span>
        <span class="badge badge-gt">Ground Truth: {'Available' if gt_available else 'Not Supplied'}</span>
      </div>
      <p style="margin-top: 0.75rem; color: var(--text-muted);">
        <strong>Document ID:</strong> {doc_id} | <strong>Filename:</strong> {filename}
      </p>
    </header>

    <div class="grid">
      <!-- OCR Section -->
      <div class="card">
        <h2>OCR Recognition</h2>
        <table>
          <tr><td>Status</td><td>{ocr.get('status', 'N/A')}</td></tr>
          <tr><td>Character Error Rate (CER)</td><td>{val_display(ocr.get('cer'))}</td></tr>
          <tr><td>Word Error Rate (WER)</td><td>{val_display(ocr.get('wer'))}</td></tr>
          <tr><td>Character Accuracy</td><td>{val_display(ocr.get('character_accuracy'))}</td></tr>
          <tr><td>Evaluated Pages</td><td>{ocr.get('total_evaluated_pages', 0)}</td></tr>
        </table>
      </div>

      <!-- Layout Section -->
      <div class="card">
        <h2>Layout Detection</h2>
        <table>
          <tr><td>Status</td><td>{layout.get('status', 'N/A')}</td></tr>
          <tr><td>Overall Precision</td><td>{val_display(layout.get('overall_precision'))}</td></tr>
          <tr><td>Overall Recall</td><td>{val_display(layout.get('overall_recall'))}</td></tr>
          <tr><td>Overall F1 Score</td><td>{val_display(layout.get('overall_f1'))}</td></tr>
          <tr><td>Mean IoU</td><td>{val_display(layout.get('mean_iou'))}</td></tr>
          <tr><td>Total Predicted Regions</td><td>{layout.get('total_predicted_regions', 0)}</td></tr>
        </table>
      </div>

      <!-- Structure Section -->
      <div class="card">
        <h2>Structure & Hierarchy</h2>
        <table>
          <tr><td>Status</td><td>{struct.get('status', 'N/A')}</td></tr>
          <tr><td>Predicted Chapters / Sections</td><td>{struct.get('total_predicted_chapters', 0)} / {struct.get('total_predicted_sections', 0)}</td></tr>
          <tr><td>Parent-Child Integrity</td><td>{val_display(struct.get('parent_child_integrity'))}</td></tr>
          <tr><td>Page Boundary Integrity</td><td>{val_display(struct.get('page_boundary_integrity'))}</td></tr>
          <tr><td>Tree Similarity (ZSS/APTED)</td><td>{val_display(struct.get('normalized_tree_similarity'))}</td></tr>
          <tr><td>Chapter Boundary F1</td><td>{val_display(struct.get('chapter_boundary_f1'))}</td></tr>
        </table>
      </div>

      <!-- Reading Order Section -->
      <div class="card">
        <h2>Reading Order</h2>
        <table>
          <tr><td>Status</td><td>{order.get('status', 'N/A')}</td></tr>
          <tr><td>Consistency Score</td><td>{val_display(order.get('reading_order_consistency_score'))}</td></tr>
          <tr><td>Pairwise Ordering Accuracy</td><td>{val_display(order.get('pairwise_ordering_accuracy'))}</td></tr>
          <tr><td>Non-monotonic Pages</td><td>{order.get('non_monotonic_pages', 0)}</td></tr>
          <tr><td>Duplicate Order Pages</td><td>{order.get('duplicate_order_pages', 0)}</td></tr>
        </table>
      </div>

      <!-- Chunks Section -->
      <div class="card">
        <h2>Chunking Integrity</h2>
        <table>
          <tr><td>Total Chunks</td><td>{chunks.get('total_chunks', 0)}</td></tr>
          <tr><td>Avg / Median Token Size</td><td>{chunks.get('average_token_size', 0)} / {chunks.get('median_token_size', 0)}</td></tr>
          <tr><td>Chunk Integrity Score</td><td>{val_display(chunks.get('chunk_integrity_score'))}</td></tr>
          <tr><td>Boundary F1</td><td>{val_display(chunks.get('boundary_f1'))}</td></tr>
          <tr><td>Orphan / Empty Chunks</td><td>{chunks.get('orphan_chunks_count', 0)} / {chunks.get('empty_chunks_count', 0)}</td></tr>
        </table>
      </div>

      <!-- TOON Section -->
      <div class="card">
        <h2>TOON Canonical Export</h2>
        <table>
          <tr><td>TOON File Found</td><td>{toon.get('toon_file_found', False)}</td></tr>
          <tr><td>Schema Validity</td><td>{toon.get('valid', False)}</td></tr>
          <tr><td>Manifest Consistency</td><td>{toon.get('manifest_consistency', False)}</td></tr>
          <tr><td>Round-trip Validation</td><td>{toon.get('round_trip_validation', False)}</td></tr>
          <tr><td>File Size (bytes)</td><td>{toon.get('file_size_bytes', 0)}</td></tr>
        </table>
      </div>
    </div>

    <div class="footer">
      Generated by Textbook OCR Evaluation Engine — Evaluation Branch
    </div>
  </div>
</body>
</html>
"""
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_file
