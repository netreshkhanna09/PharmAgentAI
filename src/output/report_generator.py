# ==============================================================
# PDF Compliance Report Generator
# ==============================================================
# PURPOSE:
#   Generates a 21 CFR Part 11 style PDF compliance report for
#   every approved pharmaceutical marketing claim. The report
#   contains a full audit trail so any FDA auditor can reconstruct
#   exactly how the claim was produced.
#
# DESIGN DECISIONS:
#   - Library: fpdf2 (pure Python, zero OS dependencies)
#     Avoids WeasyPrint's need for Cairo/Pango system libraries.
#   - Layout: Two-column key-value table for all data sections.
#     This is the standard format in pharma regulatory documents.
#   - Timestamps: Always UTC (regulated systems are tz-independent).
#   - Color coding: Green badge for PASS, clean blue section headers.
#   - Auto page-break: fpdf2 handles this automatically.
#
# SECTIONS IN THE REPORT:
#   1. Document Header (system name, run ID, timestamp)
#   2. Final Approved Claim (the deliverable -- highlighted box)
#   3. Source Clinical Data (provenance -- proves no hallucination)
#   4. Compliance Verdict (status, confidence, loops)
#   5. Drug Compliance History (pulled from episodic memory DB)
#   6. Legal Disclaimer
# ==============================================================

from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

from src.models.schemas import AgentState


# ── Text Sanitizer ──────────────────────────────────────────────
# WHY THIS EXISTS:
# Data from ClinicalTrials.gov and LLM outputs can contain Unicode
# characters that the built-in Helvetica font cannot render:
#   em-dash (\u2014), smart quotes (\u201c, \u201d), degree (\u00b0),
#   bullet (\u2022), Greek letters (\u03b1, \u03b2), etc.
#
# Helvetica is a Latin-1 (ISO-8859-1) font. Any character above
# codepoint 255 causes a crash. We sanitize ALL text before
# passing it to fpdf2 by replacing known offenders with ASCII.
#
# ALTERNATIVE: Use a Unicode font (e.g., DejaVu) with pdf.add_font().
# But that requires bundling a .ttf file. Sanitization is simpler
# for a compliance system where the text is mostly standard English.

_UNICODE_MAP = {
    '\u2014': '-',   # em dash
    '\u2013': '-',   # en dash
    '\u2018': "'",  # left single quote
    '\u2019': "'",  # right single quote
    '\u201c': '"',  # left double quote
    '\u201d': '"',  # right double quote
    '\u00b0': ' degrees',  # degree symbol
    '\u00b1': '+/-',  # plus-minus
    '\u2022': '*',   # bullet
    '\u2264': '<=',  # less-than-or-equal
    '\u2265': '>=',  # greater-than-or-equal
    '\u03b1': 'alpha',
    '\u03b2': 'beta',
    '\u03bc': 'mu',
    '\u2212': '-',   # minus sign
    '\ufffd': '?',   # replacement character (question mark box)
}


def sanitize(text: str) -> str:
    """
    Replaces known Unicode characters that Helvetica cannot render
    with ASCII-safe equivalents, then strips any remaining non-Latin-1
    characters. Safe to call on any string before passing to fpdf2.
    """
    if not text:
        return ''
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    # Final pass: encode to Latin-1 and decode back, replacing errors
    return text.encode('latin-1', errors='replace').decode('latin-1')


# ── Color Constants ─────────────────────────────────────────────
# Consistent brand palette for the report
BLUE_DARK   = (26, 82, 118)       # Header background (dark navy)
BLUE_LIGHT  = (214, 234, 248)     # Section header fill (light blue)
GREEN_DARK  = (30, 132, 73)       # PASS badge text
GREEN_LIGHT = (212, 239, 223)     # PASS badge fill
GRAY_LIGHT  = (248, 249, 250)     # Alternating table row fill
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
GRAY_DARK   = (100, 100, 100)     # Secondary text

# ── Layout Constants ────────────────────────────────────────────
LABEL_COL_W = 65   # Width (mm) of the left "label" column in tables
PAGE_MARGIN = 15   # Page margin in mm


class ComplianceReportPDF(FPDF):
    """
    Custom FPDF subclass for PharmAgentAI compliance reports.
    Overrides header() and footer() which fpdf2 calls automatically
    on every page.
    """

    def header(self):
        """
        Draws the page header: dark navy banner with white text.
        Called by fpdf2 automatically at the top of every page.
        """
        # Dark navy banner
        self.set_fill_color(*BLUE_DARK)
        self.rect(0, 0, self.w, 22, style='F')

        # Company name (left side)
        self.set_y(4)
        self.set_x(PAGE_MARGIN)
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(*WHITE)
        self.cell(100, 7, 'PharmAgentAI', border=0)

        # Document classification (right side)
        self.set_font('Helvetica', '', 8)
        self.cell(0, 7, 'REGULATORY COMPLIANCE REPORT', border=0, align='R')

        # Subtitle line
        self.set_x(PAGE_MARGIN)
        self.set_y(12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(180, 210, 240)
        self.cell(0, 5, 'Automated Pharmaceutical Claim Generation & FDA Compliance Verification System', border=0)

        # Reset text color and move below banner
        self.set_text_color(*BLACK)
        self.set_y(28)

    def footer(self):
        """
        Draws the page footer: page number and UTC generation timestamp.
        Called by fpdf2 automatically at the bottom of every page.
        """
        self.set_y(-14)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(*GRAY_DARK)

        # Left: confidentiality notice
        self.cell(80, 5, 'CONFIDENTIAL -- FOR INTERNAL USE ONLY', border=0)
        # Center: page number
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', border=0, align='C')


# ── Helper Drawing Functions ────────────────────────────────────

def _section_title(pdf: FPDF, title: str):
    """
    Draws a section header: light blue fill with bold dark text.
    Used to visually separate major sections of the report.
    """
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(*BLUE_LIGHT)
    pdf.set_text_color(*BLUE_DARK)
    pdf.cell(0, 8, f'  {title}', border=0, new_x='LMARGIN', new_y='NEXT', fill=True)
    pdf.set_text_color(*BLACK)
    pdf.ln(1)


def _table_row(pdf: FPDF, label: str, value: str, shade: bool = False):
    """
    Draws one key-value row in a data table.

    Uses two cells side-by-side:
      - Left cell (LABEL_COL_W mm): bold label
      - Right cell (remaining width): normal text, can wrap

    WHY TWO CELLS?
    fpdf2's multi_cell always starts at the left margin and fills
    the full width. To get a two-column layout, we need cell()
    for the label (fixed width, no wrap) and a right-aligned
    multi_cell for the value. But multi_cell resets X position.

    TECHNIQUE: We save the Y position before calling cell(),
    then after cell() the cursor is on the SAME line (new_y='TOP').
    We set X to LABEL_COL_W + margin, then call multi_cell for value.
    Then we set Y to whichever is further down.
    """
    fill_color = GRAY_LIGHT if shade else WHITE
    pdf.set_fill_color(*fill_color)

    start_y = pdf.get_y()
    x_left  = PAGE_MARGIN
    x_right = PAGE_MARGIN + LABEL_COL_W
    value_w = pdf.w - x_right - PAGE_MARGIN

    # Draw label cell (left column)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(*GRAY_DARK)
    pdf.set_x(x_left)
    pdf.cell(LABEL_COL_W, 7, f'  {label}', border=0,
             new_x='RIGHT', new_y='TOP', fill=True)

    # Draw value cell (right column, may wrap)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*BLACK)
    pdf.set_xy(x_right, start_y)
    pdf.multi_cell(value_w, 7, str(value), border=0, fill=True)

    # Ensure we're below both cells before next row
    end_y = pdf.get_y()
    if end_y < start_y + 7:
        pdf.set_y(start_y + 7)


def _pass_badge(pdf: FPDF):
    """Draws a green PASS status badge."""
    badge_text = '  APPROVED - COMPLIANT  '   # ASCII only — no em-dashes
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(*GREEN_LIGHT)
    pdf.set_text_color(*GREEN_DARK)
    badge_w = pdf.get_string_width(badge_text) + 6
    pdf.cell(badge_w, 10, badge_text,
             border=0, new_x='LMARGIN', new_y='NEXT', fill=True, align='C')
    pdf.set_text_color(*BLACK)


# ── Main Report Generator ───────────────────────────────────────

def generate_pdf_report(state: AgentState, run_id: str, run_stats: dict) -> str:
    """
    Generates a professional FDA compliance audit report as a PDF.

    Args:
        state:     The final AgentState from the completed pipeline run.
        run_id:    The database UUID for this run (for traceability).
        run_stats: Historical statistics from episodic memory.

    Returns:
        Absolute path to the generated PDF file.

    WHY RETURN THE PATH?
    The caller (generate_output_node) prints it to the console so
    the operator knows exactly where to find the file. In production,
    this path would also be stored in the database and/or emailed.
    """
    # ── Create Output Directory ─────────────────────────────────
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Filename: timestamped + drug name for easy sorting ──────
    # Pattern: compliance_report_pembrolizumab_20240914_165306.pdf
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_drug  = state.drug_name.replace(" ", "_").lower()
    filename   = reports_dir / f"compliance_report_{safe_drug}_{ts}.pdf"
    ts_utc     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Build PDF ───────────────────────────────────────────────
    pdf = ComplianceReportPDF(orientation='P', unit='mm', format='A4')
    pdf.alias_nb_pages()   # enables {nb} in footer
    pdf.set_margins(PAGE_MARGIN, 10, PAGE_MARGIN)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── SECTION 0: Document Metadata Bar ───────────────────────
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(*GRAY_DARK)
    pdf.cell(0, 5, f'Run ID: {run_id}   |   Generated: {ts_utc}   |   Status: APPROVED',
             border=0, new_x='LMARGIN', new_y='NEXT', align='R')
    pdf.ln(3)

    # ── SECTION 1: PASS Badge ───────────────────────────────────
    _pass_badge(pdf)
    pdf.ln(4)

    # ── SECTION 2: Final Approved Claim ─────────────────────────
    _section_title(pdf, 'FINAL APPROVED PHARMACEUTICAL CLAIM')

    # Highlighted claim text box
    pdf.set_fill_color(255, 252, 235)  # warm yellow tint
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*BLACK)
    claim_text = sanitize(state.claim_draft.claim_text)
    pdf.multi_cell(0, 7, claim_text, border=1, fill=True)
    pdf.ln(5)

    # ── SECTION 3: Source Clinical Data (Provenance) ─────────────
    _section_title(pdf, 'SOURCE CLINICAL DATA (PROVENANCE)')
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(*GRAY_DARK)
    pdf.cell(0, 5,
             'The following data was retrieved directly from ClinicalTrials.gov '
             'and used as the sole factual basis for the claim above.',
             new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)

    rows = [
        ('Drug Name',        sanitize(state.trial_data.drug_name)),
        ('NCT Identifier',   sanitize(state.trial_data.nct_id)),
        ('Study Phase',      sanitize(state.trial_data.study_phase)),
        ('Primary Endpoint', sanitize(state.trial_data.primary_endpoint)),
        ('p-value',          sanitize(str(state.trial_data.p_value))),
        ('Hazard Ratio',     sanitize(str(state.trial_data.hazard_ratio or 'Not reported'))),
        ('95% CI',           sanitize(str(state.trial_data.confidence_interval or 'Not reported'))),
        ('Patient Population', sanitize(state.trial_data.patient_population)),
    ]
    for i, (label, value) in enumerate(rows):
        _table_row(pdf, label, value, shade=(i % 2 == 0))
    pdf.ln(5)

    # ── SECTION 4: Compliance Verdict ───────────────────────────
    _section_title(pdf, 'COMPLIANCE VERDICT')
    confidence = state.compliance_result.confidence_score or 0.0
    revision_loops = state.loop_count

    compliance_rows = [
        ('Verdict',             'PASS - Claim approved for use'),
        ('AI Confidence Score', f'{confidence:.0%}'),
        ('FDA Rule Framework',  'FDA Guidance for Industry - Pharmaceutical Promotional Materials'),
        ('RAG Knowledge Base',  '5 FDA guideline documents (21 CFR, promotional standards)'),
        ('Revision Loops',      f'{revision_loops} loop(s) - '
                                + ('First-attempt approval' if revision_loops <= 1
                                   else f'{revision_loops} drafts before final approval')),
    ]
    for i, (label, value) in enumerate(compliance_rows):
        _table_row(pdf, label, value, shade=(i % 2 == 0))
    pdf.ln(5)

    # ── SECTION 5: Execution Metadata ───────────────────────────
    _section_title(pdf, 'SYSTEM EXECUTION METADATA')
    meta_rows = [
        ('Run ID',            run_id),
        ('Generation Time',   ts_utc),
        ('Drug Queried',      state.drug_name),
        ('Output Tone',       state.claim_draft.tone or 'scientific'),
        ('AI System',         'PharmAgentAI v1.0 (LangGraph + Groq)'),
        ('Compliance Engine', 'FAISS RAG + Adversarial Regulatory Agent (temp=0.0)'),
    ]
    for i, (label, value) in enumerate(meta_rows):
        _table_row(pdf, label, value, shade=(i % 2 == 0))
    pdf.ln(5)

    # ── SECTION 6: Drug Compliance History ──────────────────────
    _section_title(pdf, 'DRUG COMPLIANCE HISTORY (EPISODIC MEMORY)')
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(*GRAY_DARK)
    pdf.cell(0, 5, 'Historical performance data retrieved from the system database.',
             new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)

    total_runs      = run_stats.get('total_runs', 0)
    successful_runs = run_stats.get('successful_runs', 0)
    avg_loops       = run_stats.get('avg_loops', 0.0)
    pass_rate       = f'{(successful_runs / total_runs * 100):.0f}%' if total_runs > 0 else 'N/A'

    history_rows = [
        ('Total Runs (this drug)', str(total_runs)),
        ('Successful Runs',        str(successful_runs)),
        ('Pass Rate',              pass_rate),
        ('Avg. Revision Loops',    f'{avg_loops:.1f}'),
        ('Memory Status',          'Active - past failures inform future drafts'),
    ]
    for i, (label, value) in enumerate(history_rows):
        _table_row(pdf, label, value, shade=(i % 2 == 0))
    pdf.ln(8)

    # ── SECTION 7: Legal Disclaimer ─────────────────────────────
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(*GRAY_DARK)
    disclaimer = (
        'DISCLAIMER: This document was generated by PharmAgentAI, an AI-assisted '
        'regulatory compliance system. This report is intended for internal review '
        'purposes only and does not constitute final regulatory approval. All claims '
        'must be reviewed and signed off by a qualified Regulatory Affairs professional '
        'before external use. The AI compliance check is based on the retrieved FDA '
        'guideline documents and does not replace expert regulatory counsel.'
    )
    pdf.multi_cell(0, 4, disclaimer, border=0)

    # ── Save PDF ─────────────────────────────────────────────────
    pdf.output(str(filename))
    return str(filename.resolve())
