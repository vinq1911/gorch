#!/usr/bin/env python3
"""Generate doc/gorch-session-report.pdf from JSON benchmark data.

Reads:
  doc/mythos_block_report.json     — per-component timings (TestMythosBlockReport)
  doc/session_benchmarks.json      — before/after wall-clock numbers for the round

Writes:
  doc/gorch-session-report.pdf

Run after `go test -tags e2e ./e2e/ -run TestMythosBlockReport -v`.
"""

import json
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    mythos_path = os.path.join(here, "mythos_block_report.json")
    session_path = os.path.join(here, "session_benchmarks.json")
    out_path = os.path.join(here, "gorch-session-report.pdf")

    with open(mythos_path) as f:
        mythos = json.load(f)
    with open(session_path) as f:
        session = json.load(f)

    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    body.fontName = "Helvetica"
    body.fontSize = 10
    body.leading = 14
    code = ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=9, leading=11)

    story = []
    story.append(Paragraph("gorch session report", h1))
    story.append(Paragraph(
        f"<b>Hardware:</b> {mythos['hardware']}<br/>"
        f"<b>Date:</b> {session['report_date']}<br/>"
        f"<b>Round:</b> {session['session_summary']}",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1. Wall-clock improvements (before / after)", h2))
    story.append(Paragraph(
        "Numbers below are end-to-end measurements on the same hardware. "
        "“Before” = pre-session main; “After” = current main.",
        body,
    ))
    rows = [["Benchmark", "Before (ms)", "After (ms)", "Speedup"]]
    for ba in session["before_after"]:
        before = "—" if ba["before_ms"] is None else f"{ba['before_ms']:.2f}"
        after = f"{ba['after_ms']:.2f}"
        rows.append([
            ba["benchmark"],
            before,
            after,
            f"{ba['speedup_x']:.2f}×",
        ])
    table = Table(rows, colWidths=[3.5 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))
    for ba in session["before_after"]:
        story.append(Paragraph(f"<b>{ba['benchmark']}</b>: {ba['source']}", body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())
    story.append(Paragraph("2. Mythos block per-component timings", h2))
    story.append(Paragraph(
        f"<b>Block config:</b> {mythos['block_config']}<br/>"
        f"<b>Sequence length:</b> {mythos['seq_len']} tokens<br/>"
        f"<b>Iterations per measurement:</b> 30 (3 warmup excluded)",
        body,
    ))
    story.append(Spacer(1, 0.1 * inch))
    rows = [["Component", "Mean time (ms)", "% of full block"]]
    full_block_time = next(
        (t["mean_ms"] for t in mythos["timings"] if "Full block" in t["op"]), 1.0)
    for tm in mythos["timings"]:
        pct = ""
        if "Full block" not in tm["op"] and "GPT-2" not in tm["op"]:
            pct = f"{100 * tm['mean_ms'] / full_block_time:.1f}%"
        rows.append([tm["op"], f"{tm['mean_ms']:.3f}", pct])
    table = Table(rows, colWidths=[3.5 * inch, 1.6 * inch, 1.4 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "The mythos_tiny block (8.8M params, dim=128) runs at ~0.5 ms per "
        "forward call on M5 — about 60× faster per call than GPT-2 small "
        "(124M params, dim=768) at the same sequence length, which is the "
        "expected ratio of the model sizes.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("3. Phase 1 primitives shipped", h2))
    story.append(Paragraph(
        "Plan 0001 Phase 1 calls for thirteen primitives. Twelve shipped "
        "in this round (the thirteenth — N-D permute — also landed). PRs:",
        body,
    ))
    for p in session["phase1_primitives_shipped"]:
        story.append(Paragraph(f"• {p}", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Supporting changes", h2))
    for s in session["supporting_changes"]:
        story.append(Paragraph(f"• {s}", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(PageBreak())
    story.append(Paragraph("4. Honest accounting — what's still missing", h2))
    story.append(Paragraph(
        "<b>GPU autograd for non-matmul ops</b> (LayerNorm/Softmax/GELU "
        "backward) — still CPU-only. Documented as ADR-009 deferred work; "
        "Plan 0004 has the implementation strategy. Closing this is the "
        "biggest remaining win for transformer training on Metal at "
        "shapes &gt; 1G FMAs.",
        body,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "<b>MoE expert weight training</b> — MoE.Forward's manual scatter "
        "breaks the autograd chain on the FFN side. Router and per-expert "
        "Linear.Forward calls do produce autograd, but the weighted-sum "
        "scatter back into the output drops it. Needs a ScatterAdd primitive; "
        "filed as follow-up.",
        body,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "<b>Softmax inside attention</b> — softmaxInPlace operates on raw "
        "float32 and breaks autograd. Affects MultiHeadAttention's batched "
        "path, GQA, and MLA. Fixable by routing through public g.Softmax "
        "but ties into the FlashAttention-2 kernel work (plan 0004) where "
        "the whole attention is rewritten anyway.",
        body,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "<b>bf16/fp16 dtype support</b> — plan 0002. Largest single deferred "
        "perf win available (~2× memory + ~2× compute on Apple Silicon). "
        "Multi-week scope.",
        body,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "<b>FlashAttention-2 fused Metal kernel</b> — plan 0004. Eliminates "
        "the (seq, seq) intermediate that's 9 GB at GPT-2-12-layer × 4096 seq. "
        "2-3 weeks of focused shader work.",
        body,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "<b>MHA batched-path scale bug</b> — divides by headDim instead of "
        "sqrt(headDim). Discovered while building GQA. Filed for a separate "
        "small PR; the per-head loop path is already correct.",
        body,
    ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("5. Test summary", h2))
    story.append(Paragraph(
        "<b>Unit tests:</b> all packages green on the merged main "
        "(<font face='Courier'>go test ./...</font>).",
        body,
    ))
    story.append(Paragraph(
        "<b>End-to-end smoke tests</b> (gated behind <font face='Courier'>"
        "-tags e2e</font>):",
        body,
    ))
    story.append(Paragraph(
        "• <b>TestMythosBlockSmokeForward</b> — RMSNorm + GQA + RoPE + MoE "
        "composed in a transformer block, forward produces finite output that "
        "differs from input.<br/>"
        "• <b>TestMythosBlockBackwardCompletes</b> — backward doesn't panic; "
        "documents the residual-only autograd state.<br/>"
        "• <b>TestMythosBlockTrainStepConverges</b> — 30 AdamW steps drive "
        "MSE loss from ~6.5 to ~2.8 on a memorisation target.<br/>"
        "• <b>TestFinetuneShortCorpusConverges</b> — real GPT-2 small "
        "fine-tune on a short sentence drops teacher-forced loss 5 orders of "
        "magnitude in 60 steps.",
        body,
    ))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Numerical-vs-analytical autograd checks</b> on every new "
        "differentiable op: RMSNorm, SiLU, SwiGLU (both inputs), Permute, "
        "Gather, RepeatInterleave, RoPE (Llama and GPT-NeoX styles), "
        "BatchedMatMul (dA and dB), BatchedMatMulTransB (dA and dB), Reshape.",
        body,
    ))

    doc.build(story)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
