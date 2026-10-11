"""Contextual Diagnostic Assistant combining inspection context with engineering RAG."""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.chat import ChatMessage, ChatSession
from src.database.models.diagnosis import Diagnosis
from src.database.models.image_asset import ImageAsset
from src.database.models.inspection import Inspection
from src.database.models.plant import Plant
from src.database.models.telemetry import TelemetryRecord
from src.knowledge.document_store import KnowledgeDocumentStore

logger = logging.getLogger(__name__)


class ContextualAssistant:
    """Engineering assistant providing evidence-grounded answers over plant inspections."""

    def __init__(self, db_session: Session, doc_store: Optional[KnowledgeDocumentStore] = None):
        self.db = db_session
        self.doc_store = doc_store or KnowledgeDocumentStore()

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        plant_id: Optional[int] = None,
        inspection_id: Optional[int] = None,
        title: Optional[str] = None,
    ) -> ChatSession:
        """Retrieve existing chat session or initialize a new one."""
        if session_id:
            stmt = select(ChatSession).where(ChatSession.session_id == session_id)
            existing = self.db.scalars(stmt).first()
            if existing:
                return existing

        new_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        title_text = title or (
            f"Inspection #{inspection_id} Consultation"
            if inspection_id
            else (f"Plant #{plant_id} Operations" if plant_id else "General Solar Diagnostics")
        )

        session = ChatSession(
            session_id=new_id,
            plant_id=plant_id,
            inspection_id=inspection_id,
            title=title_text,
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def answer_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        plant_id: Optional[int] = None,
        inspection_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate evidence-grounded response and persist to conversation history."""
        session = self.get_or_create_session(session_id, plant_id, inspection_id)

        # 1. Fetch real contextual records from database
        plant_rec = self.db.get(Plant, plant_id) if plant_id else None
        insp_rec = self.db.get(Inspection, inspection_id) if inspection_id else None
        if insp_rec and not plant_rec and insp_rec.plant_id:
            plant_rec = self.db.get(Plant, insp_rec.plant_id)

        # Fetch diagnosis if linked to inspection
        diag_rec = None
        if insp_rec:
            stmt = select(Diagnosis).where(Diagnosis.inspection_id == insp_rec.id)
            diag_rec = self.db.scalars(stmt).first()

        # Fetch images linked to inspection
        images = []
        if insp_rec:
            stmt = select(ImageAsset).where(ImageAsset.inspection_id == insp_rec.id)
            images = self.db.scalars(stmt).all()

        # Fetch latest telemetry if plant available
        telemetry_rec = None
        if plant_rec:
            stmt = (
                select(TelemetryRecord)
                .where(TelemetryRecord.plant_id == plant_rec.id)
                .order_by(TelemetryRecord.timestamp.desc())
            )
            telemetry_rec = self.db.scalars(stmt).first()

        # 2. Retrieve authoritative engineering documentation via RAG
        retrieved_chunks = self.doc_store.search(query, top_k=2)

        # 3. Assemble structured, evidence-based reasoning
        citations = [
            f"{c['standard_reference']}: {c['section']} ({c['doc_title']})"
            for c in retrieved_chunks
        ]

        response_parts = []

        # Context Header
        if insp_rec:
            plant_label = plant_rec.name if plant_rec else "Plant"
            response_parts.append(
                f"### Context: Inspection `{insp_rec.code}` ({plant_label})"
            )
            response_parts.append(
                f"- **Lifecycle Status**: `{insp_rec.status}` | **Inspector**: {insp_rec.inspector_name} | **Date**: {insp_rec.inspection_date.strftime('%Y-%m-%d %H:%M UTC')}"
            )
            if insp_rec.ambient_temp_c is not None or insp_rec.irradiance_w_m2 is not None:
                response_parts.append(
                    f"- **Environmental Baseline**: Ambient Temp = {insp_rec.ambient_temp_c or 'N/A'}°C | "
                    f"Irradiance = {insp_rec.irradiance_w_m2 or 'N/A'} W/m²"
                )
            if images:
                from src.core.formatting import is_synthetic_asset
                img_desc = [
                    f"{img.image_type} (SYNTHETIC TEST IMAGE)" if is_synthetic_asset(img) else img.image_type
                    for img in images
                ]
                response_parts.append(f"- **Uploaded Media**: {len(images)} image(s) [{', '.join(img_desc)}]")

        # Diagnostic Section
        if diag_rec:
            is_confirmed = (diag_rec.review_status == "CONFIRMED")
            status_header = (
                "Verified / Confirmed Diagnostic Assessment"
                if is_confirmed
                else f"Diagnostic Assessment (Status: {diag_rec.review_status})"
            )
            response_parts.append(f"\n#### {status_header}")
            if not is_confirmed:
                response_parts.append(
                    "> ⚠️ **Engineer Review Pending**: This diagnostic assessment has not been formally certified by a human engineer. "
                    "AI hypotheses and uncorroborated classifications remain provisional."
                )

            response_parts.append(f"- **Fault Category**: `{diag_rec.fault_category}`")
            if diag_rec.canonical_fault_code:
                response_parts.append(f"- **Canonical Code**: `{diag_rec.canonical_fault_code}`")
            response_parts.append(f"- **Assigned Severity**: **{diag_rec.severity}** ({diag_rec.severity_justification})")

            conf_str = f"{diag_rec.model_confidence:.1%}" if diag_rec.model_confidence is not None else "N/A"
            if diag_rec.model_prediction:
                response_parts.append(
                    f"- **AI Model Output**: Predicted `{diag_rec.model_prediction}` (Confidence: {conf_str})"
                )
                if (diag_rec.model_confidence is not None and diag_rec.model_confidence < 0.50) or diag_rec.severity == "INCONCLUSIVE":
                    response_parts.append(
                        "  ⚠️ *Notice*: The AI vision prediction is low-confidence or uncorroborated. "
                        "Under IEC 62446-3, this is an AI hypothesis and must NOT be treated as a confirmed physical fault without electrical/EL corroboration."
                    )

            if diag_rec.power_loss_percent is not None:
                response_parts.append(
                    f"- **Estimated Output Impact**: ~{diag_rec.power_loss_percent:.1f}% module power reduction"
                )
            else:
                p_just = diag_rec.power_loss_justification or "Yield loss cannot be quantified without string I-V curve or dark EL testing."
                response_parts.append(
                    f"- **Estimated Output Impact**: Unquantified ({p_just})"
                )

            root_causes = json.loads(diag_rec.root_causes_json)
            if root_causes:
                response_parts.append("\n**Probable Root Causes (Hypotheses):**")
                for rc in root_causes:
                    response_parts.append(f"  • {rc}")

            corrective = json.loads(diag_rec.corrective_actions_json)
            if corrective:
                response_parts.append("\n**Recommended Corrective Actions:**")
                for ca in corrective:
                    response_parts.append(f"  • {ca}")

            req_meas = json.loads(diag_rec.required_measurements_json)
            if req_meas:
                response_parts.append("\n**Required Missing Measurements:**")
                for rm in req_meas:
                    response_parts.append(f"  • {rm}")

        elif insp_rec and not diag_rec:
            response_parts.append(
                "\n*Notice: No automated diagnostic assessment has been run yet for this inspection. "
                "Trigger 'Run AI Diagnosis' in the Diagnostic view to generate evidence synthesis.*"
            )

        # Telemetry Section
        if telemetry_rec:
            from src.core.formatting import is_synthetic_telemetry
            if is_synthetic_telemetry(telemetry_rec):
                response_parts.append("\n#### Corroborating Telemetry (SYNTHETIC TEST DATA)")
                response_parts.append(
                    "> ⚠️ **SYNTHETIC TEST DATA — NOT REAL PLANT MEASUREMENTS**: "
                    "The electrical telemetry points below were imported from a simulated test dataset for ingestion and pipeline validation. "
                    "They do NOT represent measured physical sensors or operational plant telemetry from field inverters."
                )
            else:
                response_parts.append("\n#### Latest Measured SCADA Telemetry")
            response_parts.append(
                f"- Recorded at: `{telemetry_rec.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}` | "
                f"Voltage: `{telemetry_rec.voltage_v or 'N/A'} V` | Current: `{telemetry_rec.current_a or 'N/A'} A` | "
                f"Power: `{telemetry_rec.power_kw or 'N/A'} kW` | Irradiance: `{telemetry_rec.irradiance_w_m2 or 'N/A'} W/m²`"
            )
        elif insp_rec:
            response_parts.append("\n#### Electrical Telemetry Status")
            response_parts.append(
                "- **Telemetry Data**: No stored string/inverter telemetry records exist for this asset in the database. "
                "In accordance with IEC standards, electrical measurements and I-V curves are NOT assumed or invented; physical field measurements must be imported to corroborate electrical performance."
            )

        # Engineering Standards Excerpts
        if retrieved_chunks:
            response_parts.append("\n#### Relevant Technical Standards & Criteria")
            for chunk in retrieved_chunks:
                response_parts.append(
                    f"> **{chunk['standard_reference']} ({chunk['section']})**:\n"
                    f"> {chunk['content']}"
                )

        # Specific Question Addressing
        q_lower = query.lower()
        if "safety" in q_lower or "hazard" in q_lower or "disconnect" in q_lower:
            response_parts.append(
                "\n⚠️ **Safety Directive (OSHA / IEC 60364-7-712)**: Never disconnect DC string cables under load. "
                "Always open the main inverter DC disconnect before decoupling MC4 leads to eliminate DC arc-flash hazards."
            )
        elif "irradiance" in q_lower or "weather" in q_lower:
            response_parts.append(
                "\n☀️ **Environmental Requirement**: In accordance with IEC 62446-3, thermal severity cannot be certified "
                "unless in-plane irradiance exceeds 600 W/m² and wind speed is below 4 m/s."
            )
        elif "confirm" in q_lower or "measurement" in q_lower or "i-v" in q_lower or "el" in q_lower or "curve" in q_lower:
            response_parts.append(
                "\n*Measurement & Corroboration Protocol*: To physically verify this anomaly, perform: "
                "(1) Dark/night Electroluminescence (EL) imaging to detect microcracks vs cell shunting; "
                "(2) String and module I-V curve tracing under irradiance > 700 W/m2 to measure shunt resistance and fill factor loss; "
                "(3) Calibrated radiometric thermography recording delta-T relative to adjacent sound cells."
            )
        elif "warranty" in q_lower or "replace" in q_lower:
            response_parts.append(
                "\n*Warranty Guidance*: Modules exhibiting hot spots with delta-T > 20°C and backsheet browning or "
                "permanent bypass diode failure typically qualify for manufacturer replacement under standard 10/25-year product warranties."
            )

        final_content = "\n".join(response_parts)

        # 4. Save dialogue turn to database
        user_msg = ChatMessage(session_id=session.id, role="user", content=query, citations_json="[]")
        assistant_msg = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=final_content,
            citations_json=json.dumps(citations),
        )
        self.db.add_all([user_msg, assistant_msg])
        self.db.commit()

        return {
            "session_id": session.session_id,
            "response": final_content,
            "citations": citations,
            "has_inspection_context": insp_rec is not None,
            "has_diagnosis_context": diag_rec is not None,
        }

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve full conversation history for a session."""
        stmt = select(ChatSession).where(ChatSession.session_id == session_id)
        session = self.db.scalars(stmt).first()
        if not session:
            return []
        return [m.to_dict() for m in session.messages]
