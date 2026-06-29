"""
Commercial PDF Report Generator.

Produces a professional, shareable report from an InsightHub pipeline result.

Sections:
  1. Cover page — query, date, confidence
  2. Executive Summary
  3. Overall Rating
  4. Top Pros & Cons
  5. Aspect Ratings with source breakdown
  6. Key Community Quotes
  7. Purchase Recommendation
  8. Methodology footnote
"""

import io
import logging
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

logger = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
C_PRIMARY    = colors.HexColor("#6366f1")   # indigo
C_ACCENT     = colors.HexColor("#22d3a0")   # teal
C_NEGATIVE   = colors.HexColor("#f87171")   # red
C_NEUTRAL    = colors.HexColor("#94a3b8")   # slate
C_BG_LIGHT   = colors.HexColor("#f8fafc")
C_TEXT       = colors.HexColor("#1e293b")
C_TEXT_LIGHT = colors.HexColor("#475569")
C_BORDER     = colors.HexColor("#e2e8f0")
C_STAR_FILL  = colors.HexColor("#f59e0b")   # amber


def _star_string(score: float, max_stars: int = 5) -> str:
    """Return a unicode star string for a 1–5 score."""
    filled = round(score)
    return "★" * filled + "☆" * (max_stars - filled)


def _confidence_colour(label: str) -> colors.Color:
    return {
        "high":         C_ACCENT,
        "medium":       C_PRIMARY,
        "low":          C_NEUTRAL,
        "insufficient": C_NEGATIVE,
    }.get(label, C_NEUTRAL)


class _StyleSheet:
    """Central typography definitions — no inline style values in drawing code."""

    def __init__(self):
        base = getSampleStyleSheet()

        self.cover_title = ParagraphStyle(
            "cover_title",
            fontSize=28, fontName="Helvetica-Bold",
            textColor=C_TEXT, alignment=TA_CENTER, leading=34,
        )
        self.cover_sub = ParagraphStyle(
            "cover_sub",
            fontSize=13, fontName="Helvetica",
            textColor=C_TEXT_LIGHT, alignment=TA_CENTER, leading=18,
        )
        self.section_header = ParagraphStyle(
            "section_header",
            fontSize=13, fontName="Helvetica-Bold",
            textColor=C_PRIMARY, spaceBefore=14, spaceAfter=6,
        )
        self.body = ParagraphStyle(
            "body",
            fontSize=10, fontName="Helvetica",
            textColor=C_TEXT, leading=15, spaceAfter=4,
        )
        self.body_light = ParagraphStyle(
            "body_light",
            fontSize=9, fontName="Helvetica",
            textColor=C_TEXT_LIGHT, leading=13, spaceAfter=3,
        )
        self.quote = ParagraphStyle(
            "quote",
            fontSize=9, fontName="Helvetica-Oblique",
            textColor=C_TEXT_LIGHT, leading=13,
            leftIndent=12, rightIndent=12, spaceAfter=6,
        )
        self.rating_big = ParagraphStyle(
            "rating_big",
            fontSize=36, fontName="Helvetica-Bold",
            textColor=C_PRIMARY, alignment=TA_CENTER,
        )
        self.stars = ParagraphStyle(
            "stars",
            fontSize=20, fontName="Helvetica",
            textColor=C_STAR_FILL, alignment=TA_CENTER, spaceAfter=4,
        )
        self.footnote = ParagraphStyle(
            "footnote",
            fontSize=7.5, fontName="Helvetica",
            textColor=C_TEXT_LIGHT, leading=11,
        )
        self.label = ParagraphStyle(
            "label",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=C_TEXT,
        )
        self.tag_positive = ParagraphStyle(
            "tag_positive",
            fontSize=9, fontName="Helvetica",
            textColor=C_ACCENT,
        )
        self.tag_negative = ParagraphStyle(
            "tag_negative",
            fontSize=9, fontName="Helvetica",
            textColor=C_NEGATIVE,
        )


_S = _StyleSheet()


def _page_header_footer(canvas, doc):
    """Draw header rule and footer on every page."""
    canvas.saveState()
    w, h = A4

    # Top rule
    canvas.setStrokeColor(C_PRIMARY)
    canvas.setLineWidth(2)
    canvas.line(15 * mm, h - 14 * mm, w - 15 * mm, h - 14 * mm)

    # Header brand
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(C_PRIMARY)
    canvas.drawString(15 * mm, h - 11 * mm, "InsightHub")

    # Footer
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_TEXT_LIGHT)
    canvas.drawString(15 * mm, 10 * mm, "InsightHub Intelligence Report  ·  insighthub.ai")
    canvas.drawRightString(w - 15 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf(payload: Dict) -> bytes:
    """
    Generate a PDF report from a pipeline payload dict.

    Returns raw PDF bytes — caller decides where to save or stream them.
    """
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=22 * mm, bottomMargin=20 * mm,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="normal",
    )
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame], onPage=_page_header_footer)
    ])

    story = []
    _build_story(story, payload)
    doc.build(story)
    return buf.getvalue()


def _build_story(story: list, payload: Dict):
    query           = payload.get("query", "")
    intent          = payload.get("intent", "GENERIC")
    summary         = payload.get("summary", "")
    ranking         = payload.get("ranking", [])
    overall         = payload.get("overall", None)
    aspects         = payload.get("aspects", {})
    quotes          = payload.get("quotes", [])
    source_counts   = payload.get("source_counts", {})
    generated_at    = datetime.now().strftime("%B %d, %Y")

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("InsightHub", _S.cover_sub))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"Intelligence Report", _S.cover_title))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f'"{query}"', _S.cover_sub))
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="60%", color=C_BORDER, hAlign="CENTER"))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f"Generated {generated_at}", _S.body_light))
    story.append(Spacer(1, 40 * mm))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", _S.section_header))
    story.append(HRFlowable(width="100%", color=C_BORDER))
    story.append(Spacer(1, 3 * mm))

    # Strip markdown from GPT summary
    clean_summary = _strip_markdown(summary)
    for para in clean_summary.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, _S.body))
    story.append(Spacer(1, 5 * mm))

    # ── Overall Rating ────────────────────────────────────────────────────────
    if overall is not None or ranking:
        story.append(Paragraph("Overall Rating", _S.section_header))
        story.append(HRFlowable(width="100%", color=C_BORDER))
        story.append(Spacer(1, 3 * mm))

        score = overall
        if score is None and ranking:
            score = ranking[0].get("overall_stars") if ranking else None

        if score is not None:
            score_f = float(score)
            story.append(Paragraph(f"{score_f:.1f} / 5", _S.rating_big))
            story.append(Paragraph(_star_string(score_f), _S.stars))

        story.append(Spacer(1, 5 * mm))

    # ── Ranking Table (for RANKING intent) ────────────────────────────────────
    if intent == "RANKING" and ranking:
        story.append(Paragraph("Rankings", _S.section_header))
        story.append(HRFlowable(width="100%", color=C_BORDER))
        story.append(Spacer(1, 3 * mm))

        table_data = [["#", "Name", "Score", "Stars", "Mentions"]]
        for i, item in enumerate(ranking[:10], 1):
            name     = item.get("name", "—")
            stars    = float(item.get("overall_stars", 0))
            mentions = item.get("mentions", 0)
            table_data.append([
                str(i),
                name,
                f"{stars:.2f}",
                _star_string(stars),
                str(mentions),
            ])

        t = Table(
            table_data,
            colWidths=[10 * mm, 65 * mm, 18 * mm, 32 * mm, 20 * mm],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  C_PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG_LIGHT, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, C_BORDER),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 6 * mm))

    # ── Aspect Ratings ────────────────────────────────────────────────────────
    if aspects:
        story.append(Paragraph("Aspect Ratings", _S.section_header))
        story.append(HRFlowable(width="100%", color=C_BORDER))
        story.append(Spacer(1, 3 * mm))

        aspect_rows = sorted(aspects.items(), key=lambda x: -x[1])
        asp_table_data = [["Aspect", "Score", "Bar", ""]]
        for asp, score_v in aspect_rows:
            score_f = float(score_v) if score_v is not None else 0.0
            pct     = int((score_f - 1) / 4 * 100)
            bar     = f"{'█' * (pct // 10)}{'░' * (10 - pct // 10)}"
            asp_table_data.append([asp, f"{score_f:.1f}", bar, _star_string(score_f)])

        ta = Table(
            asp_table_data,
            colWidths=[45 * mm, 16 * mm, 55 * mm, 25 * mm],
        )
        ta.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  C_PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG_LIGHT, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, C_BORDER),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TEXTCOLOR",   (2, 1), (2, -1), C_PRIMARY),
            ("FONTNAME",    (3, 1), (3, -1), "Helvetica"),
            ("TEXTCOLOR",   (3, 1), (3, -1), C_STAR_FILL),
        ]))
        story.append(ta)
        story.append(Spacer(1, 6 * mm))

    # ── Source Breakdown ──────────────────────────────────────────────────────
    if source_counts:
        story.append(Paragraph("Source Breakdown", _S.section_header))
        story.append(HRFlowable(width="100%", color=C_BORDER))
        story.append(Spacer(1, 3 * mm))
        total_src = sum(source_counts.values()) or 1
        src_rows  = [[src.title(), str(n), f"{n/total_src:.0%}"]
                     for src, n in sorted(source_counts.items(), key=lambda x: -x[1])]
        ts = Table([["Platform", "Comments", "Share"]] + src_rows,
                   colWidths=[50 * mm, 35 * mm, 35 * mm])
        ts.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  C_PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.5, C_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG_LIGHT, colors.white]),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ts)
        story.append(Spacer(1, 6 * mm))

    # ── Key Community Quotes ──────────────────────────────────────────────────
    display_quotes = quotes[:8] if quotes else []
    if not display_quotes and ranking:
        for item in ranking[:5]:
            for q in item.get("quotes", [])[:1]:
                display_quotes.append(q)

    if display_quotes:
        story.append(Paragraph("Key Community Voices", _S.section_header))
        story.append(HRFlowable(width="100%", color=C_BORDER))
        story.append(Spacer(1, 3 * mm))
        for q in display_quotes[:6]:
            text = q if isinstance(q, str) else str(q)
            story.append(Paragraph(
                "“" + text[:280] + "”",
                _S.quote,
            ))
        story.append(Spacer(1, 5 * mm))

    # ── Purchase Recommendation ───────────────────────────────────────────────
    story.append(Paragraph("Purchase Recommendation", _S.section_header))
    story.append(HRFlowable(width="100%", color=C_BORDER))
    story.append(Spacer(1, 3 * mm))

    rec_score = overall if overall is not None else (
        ranking[0].get("overall_stars", 3.0) if ranking else 3.0
    )
    rec_score = float(rec_score)

    if rec_score >= 4.2:
        rec_text  = "Strong Buy"
        rec_color = C_ACCENT
        rec_body  = (
            "Community sentiment is strongly positive. This product has demonstrated "
            "consistent approval across multiple sources and aspects. "
            "Recommended for most buyers."
        )
    elif rec_score >= 3.6:
        rec_text  = "Buy with Confidence"
        rec_color = C_PRIMARY
        rec_body  = (
            "Community sentiment is generally positive. The product delivers solid value "
            "for most users. Minor concerns exist in specific areas — review the aspect "
            "breakdown to ensure they don't affect your use case."
        )
    elif rec_score >= 2.8:
        rec_text  = "Consider Alternatives"
        rec_color = C_NEUTRAL
        rec_body  = (
            "Community sentiment is mixed. While some users are satisfied, a significant "
            "portion report issues. We recommend comparing against alternatives before "
            "purchasing."
        )
    else:
        rec_text  = "Not Recommended"
        rec_color = C_NEGATIVE
        rec_body  = (
            "Community sentiment is predominantly negative. The issues reported are "
            "widespread enough to caution against purchase at this time. "
            "Monitor for updates or consider alternatives."
        )

    rec_table = Table([[Paragraph(rec_text, ParagraphStyle(
        "rec", fontSize=14, fontName="Helvetica-Bold", textColor=rec_color
    ))]], colWidths=[doc_width := 170 * mm])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BG_LIGHT),
        ("BOX",        (0, 0), (-1, -1), 1.5, rec_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(rec_body, _S.body))
    story.append(Spacer(1, 8 * mm))

    # ── Methodology footnote ──────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", color=C_BORDER))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Methodology: Comments were collected from Reddit and YouTube, filtered for "
        "topical relevance using a two-stage pipeline (token overlap pre-filter + GPT "
        "semantic filter), and scored using a platform-normalised, source-quality-weighted "
        "sentiment model. Confidence tiers reflect sample size, source diversity, and "
        "sentiment consistency. This report was generated automatically by InsightHub.",
        _S.footnote,
    ))


def _strip_markdown(text: str) -> str:
    """Remove basic markdown formatting for PDF rendering."""
    import re
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",   r"\1", text)
    text = re.sub(r"`(.+?)`",     r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text.strip()
