# ==============================================================
# Action Agent — Email Notification
# ==============================================================
# PURPOSE:
#   After the LangGraph pipeline successfully generates an
#   FDA-compliant claim, this agent sends a notification email
#   to whoever requested the run (doctor, regulatory affairs team, etc.)
#
# HOW IT WORKS (Interview-ready explanation):
#   This is an "Action Agent" — a specialized agent whose job is
#   side-effects, not reasoning. It receives the final approved claim
#   and fires an SMTP email. No LLM involved — pure tool use.
#
# REAL-WORLD PATTERN:
#   In production pharma systems, every approved claim triggers:
#   1. Email to Medical Affairs team
#   2. Slack/Teams notification to Regulatory Affairs
#   3. JIRA ticket creation for review tracking
#   4. PDF upload to the document management system
#
# EMAIL SETUP OPTIONS:
#   Option A (Quick): Gmail with App Password (development)
#   Option B (Production): SendGrid/AWS SES (scalable, audit logs)
#   We implement Option A here. Switch to SendGrid for production.
# ==============================================================

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime


# ── Config (from .env) ─────────────────────────────────────────
# Add to your .env file:
#   EMAIL_SENDER=your_gmail@gmail.com
#   EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   (Gmail App Password, not your real password)
#   EMAIL_ENABLED=true

EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_ENABLED  = os.getenv("EMAIL_ENABLED", "false").lower() == "true"


def send_claim_notification(
    recipient_email: str,
    drug_name: str,
    final_claim: str,
    loop_count: int,
    run_id: str,
) -> bool:
    """
    Sends an HTML email notifying the recipient that their
    FDA-compliant claim is ready.

    Returns True if sent successfully, False otherwise.

    WHY HTML email?
    Plain text emails look unprofessional. HTML lets us format the
    claim clearly, highlight the approval status, and link to the
    PDF report — just like a real regulatory system would.
    """
    if not EMAIL_ENABLED:
        print(f"[Action Agent] Email disabled. Would have sent to: {recipient_email}")
        return False

    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("[Action Agent] EMAIL_SENDER or EMAIL_APP_PASSWORD not set in .env")
        return False

    if not recipient_email or "@" not in recipient_email:
        print("[Action Agent] No valid recipient email provided — skipping.")
        return False

    # ── Build HTML Email ────────────────────────────────────────
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M UTC")

    html_body = f"""
    <html>
    <body style="font-family: Inter, Arial, sans-serif; background: #f8f9fa; padding: 0; margin: 0;">
      <div style="max-width: 620px; margin: 40px auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.08);">
        
        <!-- Header -->
        <div style="background: #1a1f35; padding: 32px; text-align: center;">
          <div style="font-size: 28px; margin-bottom: 8px;">⚕️</div>
          <h1 style="color: white; font-size: 20px; margin: 0; font-weight: 700;">PharmAgentAI</h1>
          <p style="color: #94a3b8; font-size: 13px; margin: 4px 0 0 0;">Automated FDA Compliance Platform</p>
        </div>

        <!-- Body -->
        <div style="padding: 36px;">
          <div style="background: #d1fae5; border: 1px solid #6ee7b7; border-radius: 8px; padding: 12px 18px; margin-bottom: 24px;">
            <span style="color: #065f46; font-weight: 700; font-size: 13px;">✓ CLAIM APPROVED — FDA COMPLIANT</span>
          </div>

          <h2 style="font-size: 18px; color: #1e293b; margin: 0 0 8px 0;">Your claim for <em>{drug_name}</em> is ready</h2>
          <p style="color: #64748b; font-size: 14px; margin: 0 0 24px 0;">Generated on {timestamp} · {loop_count} review loop{'s' if loop_count != 1 else ''}</p>

          <!-- Claim Box -->
          <div style="background: #f8fafc; border-left: 4px solid #10b981; border-radius: 4px; padding: 20px; margin-bottom: 28px;">
            <p style="color: #334155; font-size: 15px; line-height: 1.7; margin: 0; font-style: italic;">
              &ldquo;{final_claim}&rdquo;
            </p>
          </div>

          <!-- How it was generated -->
          <h3 style="font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 16px 0;">Pipeline Summary</h3>
          <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 10px 0; color: #64748b;">🔬 Medical Agent</td>
              <td style="padding: 10px 0; color: #10b981; text-align: right; font-weight: 600;">Data fetched from ClinicalTrials.gov</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 10px 0; color: #64748b;">✍️ Commercial Agent</td>
              <td style="padding: 10px 0; color: #10b981; text-align: right; font-weight: 600;">Claim drafted</td>
            </tr>
            <tr>
              <td style="padding: 10px 0; color: #64748b;">⚖️ Regulatory Agent</td>
              <td style="padding: 10px 0; color: #10b981; text-align: right; font-weight: 600;">FDA compliance verified (RAG)</td>
            </tr>
          </table>

          <div style="margin-top: 28px; padding-top: 24px; border-top: 1px solid #e2e8f0; text-align: center;">
            <p style="color: #94a3b8; font-size: 12px; margin: 0;">Run ID: {run_id[:8]}... · 21 CFR Part 11 Compliant</p>
            <p style="color: #94a3b8; font-size: 12px; margin: 4px 0 0 0;">This claim was generated by AI and should be reviewed by your regulatory team before use.</p>
          </div>
        </div>
      </div>
    </body>
    </html>
    """

    # ── Send via SMTP ───────────────────────────────────────────
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"✓ FDA Claim Ready: {drug_name} — PharmAgentAI"
        msg["From"]    = f"PharmAgentAI <{EMAIL_SENDER}>"
        msg["To"]      = recipient_email

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, recipient_email, msg.as_string())

        print(f"[Action Agent] ✓ Email sent to {recipient_email} for {drug_name}")
        return True

    except Exception as e:
        print(f"[Action Agent] ✗ Email failed: {e}")
        return False
