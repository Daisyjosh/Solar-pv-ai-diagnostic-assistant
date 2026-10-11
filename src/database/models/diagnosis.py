"""Database ORM models for diagnostic assessments and engineering audit trails."""

from datetime import datetime, timezone
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inspection import Inspection


class Diagnosis(Base):
    """Structured diagnostic assessment record generated for an inspection."""

    __tablename__ = "diagnoses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    inspection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("inspections.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    assessment_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    fault_category: Mapped[str] = mapped_column(String(128), nullable=False)
    canonical_fault_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="NORMAL", index=True)
    severity_justification: Mapped[str] = mapped_column(Text, nullable=False)

    model_prediction: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_loss_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power_loss_justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Serialized JSON structures
    root_causes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    corrective_actions_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    preventive_actions_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    safety_precautions_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    required_measurements_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    diagnostic_limitations_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    raw_assessment_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    # Human-in-the-loop review workflow
    review_status: Mapped[str] = mapped_column(String(32), default="PENDING_REVIEW", nullable=False, index=True)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    engineer_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    inspection: Mapped["Inspection"] = relationship("Inspection", backref="diagnosis")
    audit_logs: Mapped[List["DiagnosticAuditLog"]] = relationship(
        "DiagnosticAuditLog",
        back_populates="diagnosis",
        cascade="all, delete-orphan",
        order_by="DiagnosticAuditLog.timestamp.desc()",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnosis entity to structured dictionary."""
        return {
            "id": self.id,
            "inspection_id": self.inspection_id,
            "assessment_id": self.assessment_id,
            "fault_category": self.fault_category,
            "canonical_fault_code": self.canonical_fault_code,
            "severity": self.severity,
            "severity_justification": self.severity_justification,
            "model_prediction": self.model_prediction,
            "model_confidence": self.model_confidence,
            "power_loss_percent": self.power_loss_percent,
            "power_loss_justification": self.power_loss_justification,
            "root_causes": json.loads(self.root_causes_json),
            "corrective_actions": json.loads(self.corrective_actions_json),
            "preventive_actions": json.loads(self.preventive_actions_json),
            "safety_precautions": json.loads(self.safety_precautions_json),
            "required_measurements": json.loads(self.required_measurements_json),
            "diagnostic_limitations": json.loads(self.diagnostic_limitations_json),
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "engineer_notes": self.engineer_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class DiagnosticAuditLog(Base):
    """Audit log tracking human engineer reviews and status modifications."""

    __tablename__ = "diagnostic_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    diagnosis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("diagnoses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    changed_by: Mapped[str] = mapped_column(String(128), nullable=False)
    previous_status: Mapped[str] = mapped_column(String(32), nullable=False)
    new_status: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_severity: Mapped[str] = mapped_column(String(32), nullable=False)
    new_severity: Mapped[str] = mapped_column(String(32), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    diagnosis: Mapped["Diagnosis"] = relationship("Diagnosis", back_populates="audit_logs")
