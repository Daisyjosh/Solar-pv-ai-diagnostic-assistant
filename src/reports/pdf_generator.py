"""Solar PV Professional Engineering Inspection & Diagnostic PDF Report Generator.

Uses ReportLab to compile comprehensive, audit-ready engineering inspection dossiers
containing asset hierarchy, thermal imagery, model telemetry, evidence synthesis,
standards traceability (IEC 62446-3, IEC 61215, IEA-PVPS Task 13, OSHA 1910.303),
and engineer sign-off blocks.
"""

from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.core.formatting import format_power_loss, is_synthetic_asset, is_synthetic_telemetry

logger = logging.getLogger(__name__)


class InspectionReportPDFGenerator:
    """Generates standardized IEC 62446-3 compliant PDF engineering inspection dossiers."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Configure typography and palette for solar engineering documents."""
        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Heading1"],
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#1A365D"),
            fontName="Helvetica-Bold",
            spaceAfter=3,
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=self.styles["Normal"],
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#4A5568"),
            fontName="Helvetica",
            spaceAfter=6,
        )
        self.section_heading = ParagraphStyle(
            "SectionHeading",
            parent=self.styles["Heading2"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#2B6CB0"),
            fontName="Helvetica-Bold",
            spaceBefore=6,
            spaceAfter=4,
        )
        self.body_style = ParagraphStyle(
            "Body",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#2D3748"),
            fontName="Helvetica",
        )
        self.bullet_style = ParagraphStyle(
            "BulletText",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#2D3748"),
            fontName="Helvetica",
            leftIndent=10,
        )
        self.meta_label = ParagraphStyle(
            "MetaLabel",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#718096"),
            fontName="Helvetica-Bold",
        )
        self.meta_value = ParagraphStyle(
            "MetaValue",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1A202C"),
            fontName="Helvetica",
        )
        self.warning_text = ParagraphStyle(
            "WarningText",
            parent=self.styles["Normal"],
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#9B2C2C"),
            fontName="Helvetica",
        )

    def generate(
        self,
        inspection_data: Dict[str, Any],
        diagnosis_data: Optional[Dict[str, Any]] = None,
        images_data: Optional[List[Dict[str, Any]]] = None,
        telemetry_data: Optional[List[Dict[str, Any]]] = None,
    ) -> bytes:
        """Compile inspection data into PDF binary stream."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=32,
            bottomMargin=32,
        )

        story = []

        # 1. Document Header Banner
        status_val = inspection_data.get("status", "SCHEDULED")
        header_table = Table(
            [
                [
                    Paragraph("☀️ SOLAR PV AI DIAGNOSTIC ASSISTANT", self.title_style),
                    Paragraph(
                        f"REPORT DATE: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br/>"
                        f"LIFECYCLE STATUS: <b>{status_val}</b>",
                        ParagraphStyle("HRight", parent=self.meta_label, alignment=TA_RIGHT),
                    ),
                ],
                [
                    Paragraph(
                        "IEC 62446-3 Technical Thermography & Electrical Corroboration Dossier",
                        self.subtitle_style,
                    ),
                    Paragraph(
                        f"INSPECTION: <b>{inspection_data.get('code')}</b>",
                        ParagraphStyle("HRight2", parent=self.meta_label, alignment=TA_RIGHT),
                    ),
                ],
            ],
            colWidths=[360, 180],
        )
        header_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
            ])
        )
        story.append(header_table)
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

        # 2. Plant and Equipment Asset Details Table
        story.append(Paragraph("1. Asset Identification & Environmental Metadata", self.section_heading))
        asset_info = inspection_data.get("asset_info", {})
        meta_grid = [
            [
                Paragraph("Solar Plant:", self.meta_label),
                Paragraph(str(asset_info.get("plant_name", "N/A")), self.meta_value),
                Paragraph("DC Capacity:", self.meta_label),
                Paragraph(f"{asset_info.get('capacity_dc_mw', 'N/A')} MWp", self.meta_value),
            ],
            [
                Paragraph("Inverter:", self.meta_label),
                Paragraph(str(asset_info.get("inverter_code", "Unspecified")), self.meta_value),
                Paragraph("PV String:", self.meta_label),
                Paragraph(str(asset_info.get("string_code", "Unspecified")), self.meta_value),
            ],
            [
                Paragraph("PV Module:", self.meta_label),
                Paragraph(str(asset_info.get("module_code", "Unspecified")), self.meta_value),
                Paragraph("Inspector:", self.meta_label),
                Paragraph(str(inspection_data.get("inspector_name", "N/A")), self.meta_value),
            ],
            [
                Paragraph("Inspection Date:", self.meta_label),
                Paragraph(str(inspection_data.get("inspection_date", "N/A")), self.meta_value),
                Paragraph("Ambient / Irradiance:", self.meta_label),
                Paragraph(
                    f"{inspection_data.get('ambient_temp_c', 'N/A')} °C | {inspection_data.get('irradiance_w_m2', 'N/A')} W/m²",
                    self.meta_value,
                ),
            ],
        ]
        asset_table = Table(meta_grid, colWidths=[95, 175, 100, 170])
        asset_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ])
        )
        story.append(asset_table)
        story.append(Spacer(1, 6))

        # 3. Diagnostic Assessment & AI Vision Synthesis
        story.append(Paragraph("2. Diagnostic Synthesis & Severity Classification", self.section_heading))

        if diagnosis_data:
            severity = diagnosis_data.get("severity", "NORMAL")
            sev_color = {
                "CRITICAL": colors.HexColor("#E53E3E"),
                "MAJOR": colors.HexColor("#DD6B20"),
                "MINOR": colors.HexColor("#D69E2E"),
                "NORMAL": colors.HexColor("#38A169"),
                "INCONCLUSIVE": colors.HexColor("#718096"),
            }.get(severity, colors.HexColor("#718096"))

            conf_val = diagnosis_data.get("model_confidence")
            conf_str = f"{conf_val:.1%}" if conf_val is not None else "N/A"
            pred_str = diagnosis_data.get("model_prediction") or "N/A"
            ploss_str = format_power_loss(diagnosis_data.get("power_loss_percent"), fallback="Not quantified")
            review_status = diagnosis_data.get("review_status", "PENDING_REVIEW")

            # Structured Grid - 4 columns totaling 540 pt
            diag_grid = [
                [
                    Paragraph("Fault Classification:", self.meta_label),
                    Paragraph(
                        f"<b>{diagnosis_data.get('fault_category')}</b> ({diagnosis_data.get('canonical_fault_code', 'Uncoded')})",
                        self.meta_value,
                    ),
                    Paragraph("Assigned Severity:", self.meta_label),
                    Paragraph(f"<font color='{sev_color.hexval()}'><b>{severity}</b></font>", self.meta_value),
                ],
                [
                    Paragraph("AI Model Prediction:", self.meta_label),
                    Paragraph(f"{pred_str} (Confidence: {conf_str})", self.meta_value),
                    Paragraph("Est. Yield Loss:", self.meta_label),
                    Paragraph(ploss_str, self.meta_value),
                ],
                [
                    Paragraph("Review Status:", self.meta_label),
                    Paragraph(f"<b>{review_status}</b>", self.meta_value),
                    Paragraph("Verification Need:", self.meta_label),
                    Paragraph(
                        "I-V curve & EL test required" if severity == "INCONCLUSIVE" else "Visual & Thermal Match",
                        self.meta_value,
                    ),
                ],
            ]
            diag_table = Table(diag_grid, colWidths=[105, 165, 105, 165])
            diag_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ])
            )
            story.append(diag_table)
            story.append(Spacer(1, 4))

            # Full-Width Severity Rationale Callout Box (540 pt width)
            sev_just = diagnosis_data.get("severity_justification", "N/A")
            rationale_box = Table(
                [
                    [
                        Paragraph(
                            f"<b>Engineering Severity Rationale & Criteria:</b> {sev_just}",
                            self.body_style,
                        )
                    ]
                ],
                colWidths=[540],
            )
            rationale_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(rationale_box)
            story.append(Spacer(1, 5))

            # Hypotheses & Corrective Actions
            root_causes = diagnosis_data.get("root_causes", [])
            if root_causes:
                story.append(Paragraph("<b>Probable Root Causes (Engineering Hypotheses):</b>", self.body_style))
                for rc in root_causes:
                    story.append(Paragraph(f"• {rc}", self.bullet_style))
                story.append(Spacer(1, 3))

            corrective = diagnosis_data.get("corrective_actions", [])
            if corrective:
                story.append(Paragraph("<b>Recommended Corrective & Preventive Measures:</b>", self.body_style))
                for ca in corrective:
                    story.append(Paragraph(f"• {ca}", self.bullet_style))
                story.append(Spacer(1, 3))

            req_meas = diagnosis_data.get("required_measurements", [])
            if req_meas:
                story.append(Paragraph("<b>Required Missing Electrical Measurements:</b>", self.body_style))
                for rm in req_meas:
                    story.append(Paragraph(f"• {rm}", self.bullet_style))
                story.append(Spacer(1, 3))
        else:
            story.append(
                Paragraph(
                    "<i>No formal AI diagnostic assessment has been computed for this inspection record. "
                    "Trigger the automated diagnostic engine to generate fault classification.</i>",
                    self.body_style,
                )
            )

        story.append(Spacer(1, 5))

        # 4. Images Section (Embedded thumbnails if available)
        story.append(Paragraph("3. Visual & Thermographic Evidence Assets", self.section_heading))
        if images_data:
            img_rows = []
            for img_info in images_data[:4]:  # Show up to 4 images
                raw_path = img_info.get("file_path", "")
                is_synthetic = is_synthetic_asset(img_info)

                cell_flowables = [
                    Paragraph(f"<b>{img_info.get('image_type')}</b>", self.meta_label),
                    Paragraph(f"File: {img_info.get('original_filename')}", self.body_style),
                ]

                if is_synthetic:
                    cell_flowables.append(
                        Paragraph(
                            "<font color='#C53030'><b>[SYNTHETIC TEST IMAGE — BENCHMARK/SIMULATED]</b></font>",
                            ParagraphStyle("SynImgLabel", parent=self.meta_label, fontSize=7, leading=9, textColor=colors.HexColor("#C53030")),
                        )
                    )

                resolved_img_path = None
                if raw_path:
                    try:
                        from src.services.storage import StorageService
                        resolved_img_path = StorageService().get_image_file_path(raw_path)
                    except Exception:
                        p = Path(raw_path)
                        if p.exists() and p.is_file():
                            resolved_img_path = p

                if resolved_img_path and resolved_img_path.exists():
                    try:
                        rl_img = RLImage(str(resolved_img_path), width=96, height=76)
                        cell_flowables.insert(0, rl_img)
                    except Exception as e:
                        logger.warning("Could not embed image %s in PDF: %s", resolved_img_path, e)
                img_rows.append(cell_flowables)

            # Format in paired 2-column layout (270 pt + 270 pt = 540 pt)
            if img_rows:
                paired_rows = []
                for i in range(0, len(img_rows), 2):
                    row_cells = img_rows[i : i + 2]
                    if len(row_cells) == 1:
                        row_cells.append([Paragraph("", self.body_style)])
                    paired_rows.append(row_cells)

                img_table = Table(paired_rows, colWidths=[270, 270])
                img_table.setStyle(
                    TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ])
                )
                story.append(img_table)
        else:
            story.append(Paragraph("<i>No image assets linked to this inspection.</i>", self.body_style))

        story.append(Spacer(1, 6))

        # 5. Electrical Telemetry Table
        story.append(Paragraph("4. Corroborating Telemetry & Operating Points", self.section_heading))

        has_synthetic_telemetry = any(is_synthetic_telemetry(tr) for tr in (telemetry_data or []))
        if has_synthetic_telemetry:
            # Prominent Synthetic Data Warning Callout Box
            syn_box = Table(
                [
                    [
                        Paragraph(
                            "<b><font color='#9B2C2C'>⚠️ SYNTHETIC TEST DATA — NOT REAL PLANT MEASUREMENTS</font></b><br/>"
                            "The electrical operating points below were imported from a simulated test dataset for ingestion and pipeline validation. "
                            "They do NOT represent measured physical sensors or operational SCADA telemetry from this solar plant.",
                            self.warning_text,
                        )
                    ]
                ],
                colWidths=[540],
            )
            syn_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF5F5")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#FEB2B2")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(syn_box)
            story.append(Spacer(1, 4))

        if telemetry_data:
            t_rows = [
                [
                    Paragraph("Timestamp (UTC)", self.meta_label),
                    Paragraph("Voltage (V)", self.meta_label),
                    Paragraph("Current (A)", self.meta_label),
                    Paragraph("Power (kW)", self.meta_label),
                    Paragraph("Irradiance (W/m²)", self.meta_label),
                    Paragraph("Module Temp (°C)", self.meta_label),
                ]
            ]
            for tr in telemetry_data[:6]:
                t_rows.append([
                    Paragraph(str(tr.get("timestamp", ""))[:19].replace("T", " "), self.body_style),
                    Paragraph(f"{tr.get('voltage_v', 'N/A')}", self.body_style),
                    Paragraph(f"{tr.get('current_a', 'N/A')}", self.body_style),
                    Paragraph(f"{tr.get('power_kw', 'N/A')}", self.body_style),
                    Paragraph(f"{tr.get('irradiance_w_m2', 'N/A')}", self.body_style),
                    Paragraph(f"{tr.get('module_temp_c', 'N/A')}", self.body_style),
                ])
            tel_table = Table(t_rows, colWidths=[120, 80, 80, 80, 95, 85])
            tel_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ])
            )
            story.append(tel_table)
        else:
            story.append(
                Paragraph(
                    "<i>No time-correlated telemetry points imported. Telemetry import via CSV/Excel "
                    "is recommended to corroborate visual findings with electrical operating data.</i>",
                    self.body_style,
                )
            )

        story.append(Spacer(1, 6))

        # 6. Applicable Standards & Traceability
        story.append(Paragraph("5. Applicable Technical Standards & Inspection Norms", self.section_heading))
        standards_text = (
            "• <b>IEC 62446-3 Clause 5.1 & 5.2.2</b>: Outdoor IR thermography baseline (irradiance &ge; 600 W/m², wind &lt; 4 m/s). "
            "Class 2 Major anomaly (&Delta;T 10–20°C); Class 3 Critical anomaly (&Delta;T &gt; 20°C or backsheet browning).<br/>"
            "• <b>IEC 62446-3 Clause 5.2.3</b>: Bypass diode substring heating &sim;33.3% voltage reduction per conducting diode.<br/>"
            "• <b>IEC 61215 MQT 10.9</b>: Hot-spot endurance qualification. Electroluminescence (EL) is mandatory to map inactive wafer fragments.<br/>"
            "• <b>IEA-PVPS Task 13 Section 3.2</b>: Partial shading, reverse-bias cell dissipation, and EVA encapsulant degradation.<br/>"
            "• <b>OSHA 1910.303 / IEC 60364-7-712</b>: High-voltage DC arc-flash prevention. Never disconnect MC4 connectors under load."
        )
        std_box = Table([[Paragraph(standards_text, self.body_style)]], colWidths=[540])
        std_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(std_box)
        story.append(Spacer(1, 8))

        # 7. Engineer Review & Sign-Off Section (Wrapped in KeepTogether to prevent bad page split)
        is_pending = not diagnosis_data or diagnosis_data.get("review_status") == "PENDING_REVIEW"
        is_confirmed = bool(diagnosis_data and diagnosis_data.get("review_status") == "CONFIRMED")
        is_rejected = bool(diagnosis_data and diagnosis_data.get("review_status") == "REJECTED")
        is_modified = bool(diagnosis_data and diagnosis_data.get("review_status") == "MODIFIED")

        reviewer_name = diagnosis_data.get("reviewed_by") if (diagnosis_data and not is_pending and diagnosis_data.get("reviewed_by")) else "_________________"
        reviewed_at = diagnosis_data.get("reviewed_at", "")[:10] if (diagnosis_data and not is_pending and diagnosis_data.get("reviewed_at")) else "___________"

        sign_grid = [
            [
                Paragraph("Inspecting Technician:", self.meta_label),
                Paragraph("Lead Reviewing Engineer:", self.meta_label),
                Paragraph("Final Engineering Decision:", self.meta_label),
            ],
            [
                Paragraph(
                    f"Name: {inspection_data.get('inspector_name', '_________________')}<br/>"
                    f"Signature: __________________<br/>"
                    f"Date: {inspection_data.get('inspection_date', '')[:10] or '___________'}",
                    self.body_style,
                ),
                Paragraph(
                    f"Name: {reviewer_name}<br/>"
                    f"Signature: {'[Signed Electronically]' if not is_pending else '__________________'}<br/>"
                    f"Date: {reviewed_at}",
                    self.body_style,
                ),
                Paragraph(
                    f"[ {'X' if is_confirmed else ' '} ] Confirmed Defect<br/>"
                    f"[ {'X' if is_rejected else ' '} ] False Alarm / Normal<br/>"
                    f"[ {'X' if is_modified else ' '} ] Modified Hypothesis<br/>"
                    f"<i>{'[Pending Engineer Verification]' if is_pending else '[Review Completed]'}</i>",
                    self.body_style,
                ),
            ],
        ]
        sign_table = Table(sign_grid, colWidths=[180, 180, 180])
        sign_table.setStyle(
            TableStyle([
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2B6CB0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )

        footer_note = Paragraph(
            "<b>Engineering Disclaimer</b>: This diagnostic dossier combines automated machine vision screening with rule-based multi-modal reasoning. "
            "AI predictions represent statistical hypotheses and must be physically verified by qualified high-voltage personnel before module de-stringing or replacement.",
            ParagraphStyle("FooterNote", parent=self.styles["Italic"], fontSize=7, leading=9, textColor=colors.HexColor("#718096")),
        )

        sign_block = KeepTogether([
            Paragraph("6. Operations & Maintenance Sign-Off", self.section_heading),
            sign_table,
            Spacer(1, 6),
            footer_note,
        ])
        story.append(sign_block)

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
