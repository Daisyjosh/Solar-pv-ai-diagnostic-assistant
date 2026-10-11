"""Add diagnosis, telemetry, chat, and batch inspection tables.

Revision ID: 0002_full_platform_schema
Revises: 0001_initial_schema
Create Date: 2026-10-10 19:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0002_full_platform_schema"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Diagnoses Table
    op.create_table(
        "diagnoses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("inspection_id", sa.Integer(), nullable=False),
        sa.Column("assessment_id", sa.String(length=64), nullable=False),
        sa.Column("fault_category", sa.String(length=128), nullable=False),
        sa.Column("canonical_fault_code", sa.String(length=64), nullable=True),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("severity_justification", sa.Text(), nullable=False),
        sa.Column("model_prediction", sa.String(length=64), nullable=True),
        sa.Column("model_confidence", sa.Float(), nullable=True),
        sa.Column("power_loss_percent", sa.Float(), nullable=True),
        sa.Column("power_loss_justification", sa.Text(), nullable=True),
        sa.Column("root_causes_json", sa.Text(), nullable=False),
        sa.Column("corrective_actions_json", sa.Text(), nullable=False),
        sa.Column("preventive_actions_json", sa.Text(), nullable=False),
        sa.Column("safety_precautions_json", sa.Text(), nullable=False),
        sa.Column("required_measurements_json", sa.Text(), nullable=False),
        sa.Column("diagnostic_limitations_json", sa.Text(), nullable=False),
        sa.Column("raw_assessment_json", sa.Text(), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("reviewed_by", sa.String(length=128), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("engineer_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["inspection_id"], ["inspections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_diagnoses_id"), "diagnoses", ["id"], unique=False)
    op.create_index(op.f("ix_diagnoses_inspection_id"), "diagnoses", ["inspection_id"], unique=True)
    op.create_index(op.f("ix_diagnoses_assessment_id"), "diagnoses", ["assessment_id"], unique=True)
    op.create_index(op.f("ix_diagnoses_severity"), "diagnoses", ["severity"], unique=False)
    op.create_index(op.f("ix_diagnoses_review_status"), "diagnoses", ["review_status"], unique=False)

    # 2. Diagnostic Audit Logs Table
    op.create_table(
        "diagnostic_audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("diagnosis_id", sa.Integer(), nullable=False),
        sa.Column("changed_by", sa.String(length=128), nullable=False),
        sa.Column("previous_status", sa.String(length=32), nullable=False),
        sa.Column("new_status", sa.String(length=32), nullable=False),
        sa.Column("previous_severity", sa.String(length=32), nullable=False),
        sa.Column("new_severity", sa.String(length=32), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["diagnosis_id"], ["diagnoses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_diagnostic_audit_logs_id"), "diagnostic_audit_logs", ["id"], unique=False)
    op.create_index(op.f("ix_diagnostic_audit_logs_diagnosis_id"), "diagnostic_audit_logs", ["diagnosis_id"], unique=False)

    # 3. Telemetry Batches Table
    op.create_table(
        "telemetry_batches",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plant_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=32), nullable=False),
        sa.Column("records_count", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("import_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["plant_id"], ["plants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_telemetry_batches_id"), "telemetry_batches", ["id"], unique=False)
    op.create_index(op.f("ix_telemetry_batches_plant_id"), "telemetry_batches", ["plant_id"], unique=False)

    # 4. Telemetry Records Table
    op.create_table(
        "telemetry_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("batch_id", sa.Integer(), nullable=False),
        sa.Column("plant_id", sa.Integer(), nullable=False),
        sa.Column("inverter_id", sa.Integer(), nullable=True),
        sa.Column("string_id", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("voltage_v", sa.Float(), nullable=True),
        sa.Column("current_a", sa.Float(), nullable=True),
        sa.Column("power_kw", sa.Float(), nullable=True),
        sa.Column("irradiance_w_m2", sa.Float(), nullable=True),
        sa.Column("ambient_temp_c", sa.Float(), nullable=True),
        sa.Column("module_temp_c", sa.Float(), nullable=True),
        sa.Column("operating_status", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["telemetry_batches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["plant_id"], ["plants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["inverter_id"], ["inverters.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["string_id"], ["strings.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_telemetry_records_id"), "telemetry_records", ["id"], unique=False)
    op.create_index(op.f("ix_telemetry_records_batch_id"), "telemetry_records", ["batch_id"], unique=False)
    op.create_index(op.f("ix_telemetry_records_plant_id"), "telemetry_records", ["plant_id"], unique=False)
    op.create_index(op.f("ix_telemetry_records_timestamp"), "telemetry_records", ["timestamp"], unique=False)

    # 5. Chat Sessions Table
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("plant_id", sa.Integer(), nullable=True),
        sa.Column("inspection_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["plant_id"], ["plants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["inspection_id"], ["inspections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_sessions_id"), "chat_sessions", ["id"], unique=False)
    op.create_index(op.f("ix_chat_sessions_session_id"), "chat_sessions", ["session_id"], unique=True)

    # 6. Chat Messages Table
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_messages_id"), "chat_messages", ["id"], unique=False)
    op.create_index(op.f("ix_chat_messages_session_id"), "chat_messages", ["session_id"], unique=False)

    # 7. Batch Inspections Table
    op.create_table(
        "batch_inspections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("plant_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("total_images", sa.Integer(), nullable=False),
        sa.Column("processed_images", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["plant_id"], ["plants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_batch_inspections_id"), "batch_inspections", ["id"], unique=False)
    op.create_index(op.f("ix_batch_inspections_code"), "batch_inspections", ["code"], unique=True)


def downgrade() -> None:
    op.drop_table("batch_inspections")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("telemetry_records")
    op.drop_table("telemetry_batches")
    op.drop_table("diagnostic_audit_logs")
    op.drop_table("diagnoses")
