# ==============================================================
# PDF Report Generator
# ==============================================================
# PURPOSE: Generates a 21 CFR Part 11 compliant-style PDF report
#          containing the final approved claim, source data, and
#          a full audit trail of the agent's execution.
#
# WHY PDF?
# In regulated industries, output must be immutable and archivable.
# JSON is for machines; PDF is for human auditors.
#
# LIBRARY: fpdf2
# We use a code-based PDF generator rather than HTML/CSS templates
# to keep deployment simple (no heavy external system dependencies).
# ==============================================================

import os
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from src.models.schemas import AgentState


class ComplianceReportPDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'PharmAgentAI - Regulatory Compliance Report', 0, 1, 'C')
        # Line break
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        # Timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.cell(0, 10, f'Generated: {timestamp}', 0, 0, 'R')


def generate_pdf_report(state: AgentState, run_id: str, run_stats: dict) -> str:
    """
    Generates a PDF compliance report for an approved claim.
    Returns the absolute path to the generated PDF.
    """
    # Ensure output directory exists
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_drug_name = state.drug_name.replace(" ", "_").lower()
    filename = reports_dir / f"compliance_report_{safe_drug_name}_{timestamp}.pdf"

    pdf = ComplianceReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Helper function for sections
    def section_title(title: str):
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(0, 8, f' {title} ', 0, 1, 'L', fill=True)
        pdf.ln(2)

    def row(label: str, value: str):
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, label, 0, 1)  # ln=1 moves to next line
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, str(value))
        pdf.ln(2)

    # --- 1. FINAL APPROVED CLAIM ---
    section_title("FINAL APPROVED CLAIM")
    pdf.set_font('Arial', 'B', 11)
    pdf.multi_cell(0, 6, state.claim_draft.claim_text)
    pdf.ln(5)

    # --- 2. SOURCE DATA (Audit Trail) ---
    section_title("SOURCE CLINICAL DATA (PROVENANCE)")
    row("Drug Name:", state.trial_data.drug_name)
    row("NCT ID:", state.trial_data.nct_id)
    row("Study Phase:", state.trial_data.study_phase)
    row("Primary Endpoint:", state.trial_data.primary_endpoint)
    row("p-value:", str(state.trial_data.p_value))
    row("Hazard Ratio:", str(state.trial_data.hazard_ratio or "Not reported"))
    row("95% CI:", str(state.trial_data.confidence_interval or "Not reported"))
    row("Patient Pop.:", state.trial_data.patient_population)
    pdf.ln(5)

    # --- 3. COMPLIANCE METADATA ---
    section_title("COMPLIANCE VERDICT")
    row("Status:", "PASS")
    row("Confidence Score:", f"{state.compliance_result.confidence_score:.2f}")
    pdf.ln(5)

    # --- 4. EXECUTION METADATA ---
    section_title("SYSTEM METADATA")
    row("Run ID:", run_id)
    row("Revision Loops:", str(state.loop_count))
    row("Target Tone:", state.claim_draft.tone)
    pdf.ln(5)

    # --- 5. DRUG HISTORY (Episodic Memory Stats) ---
    section_title("DRUG COMPLIANCE HISTORY")
    row("Total Runs:", str(run_stats.get("total_runs", 0)))
    row("Successful Runs:", str(run_stats.get("successful_runs", 0)))
    row("Avg. Loops/Run:", str(run_stats.get("avg_loops", 0.0)))
    
    # Save the PDF
    pdf.output(str(filename))
    
    return str(filename.absolute())
