"""Initial database schema for solar PV assets, inspections, and media.

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-10-10 15:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Plants Table
    op.create_table(
        "plants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("location", sa.String(length=255), nullable=False),
        sa.Column("capacity_dc_mw", sa.Float(), nullable=False),
        sa.Column("commissioning_date", sa.Date(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_plants_code"), "plants", ["code"], unique=True)
    op.create_index(op.f("ix_plants_id"), "plants", ["id"], unique=False)
    op.create_index(op.f("ix_plants_name"), "plants", ["name"], unique=False)

    # 2. Inverters Table
    op.create_table(
        "inverters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plant_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("rated_power_kw", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["plant_id"], ["plants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_inverters_code"), "inverters", ["code"], unique=False)
    op.create_index(op.f("ix_inverters_id"), "inverters", ["id"], unique=False)
    op.create_index(op.f("ix_inverters_plant_id"), "inverters", ["plant_id"], unique=False)

    # 3. Strings Table
    op.create_table(
        "strings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("inverter_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("modules_count", sa.Integer(), nullable=False),
        sa.Column("rated_voc_v", sa.Float(), nullable=False),
        sa.Column("rated_isc_a", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["inverter_id"], ["inverters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_strings_code"), "strings", ["code"], unique=False)
    op.create_index(op.f("ix_strings_id"), "strings", ["id"], unique=False)
    op.create_index(op.f("ix_strings_inverter_id"), "strings", ["inverter_id"], unique=False)

    # 4. Modules Table
    op.create_table(
        "modules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("string_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("position_in_string", sa.Integer(), nullable=False),
        sa.Column("manufacturer", sa.String(length=128), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("rated_power_w", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["string_id"], ["strings.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_modules_code"), "modules", ["code"], unique=False)
    op.create_index(op.f("ix_modules_id"), "modules", ["id"], unique=False)
    op.create_index(op.f("ix_modules_string_id"), "modules", ["string_id"], unique=False)

    # 5. Inspections Table
    op.create_table(
        "inspections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("plant_id", sa.Integer(), nullable=False),
        sa.Column("inverter_id", sa.Integer(), nullable=True),
        sa.Column("string_id", sa.Integer(), nullable=True),
        sa.Column("module_id", sa.Integer(), nullable=True),
        sa.Column("inspector_name", sa.String(length=128), nullable=False),
        sa.Column("inspection_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("ambient_temp_c", sa.Float(), nullable=True),
        sa.Column("irradiance_w_m2", sa.Float(), nullable=True),
        sa.Column("weather_notes", sa.Text(), nullable=True),
        sa.Column("general_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["inverter_id"], ["inverters.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["module_id"], ["modules.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["plant_id"], ["plants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["string_id"], ["strings.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_inspections_code"), "inspections", ["code"], unique=True)
    op.create_index(op.f("ix_inspections_id"), "inspections", ["id"], unique=False)
    op.create_index(op.f("ix_inspections_inverter_id"), "inspections", ["inverter_id"], unique=False)
    op.create_index(op.f("ix_inspections_module_id"), "inspections", ["module_id"], unique=False)
    op.create_index(op.f("ix_inspections_plant_id"), "inspections", ["plant_id"], unique=False)
    op.create_index(op.f("ix_inspections_string_id"), "inspections", ["string_id"], unique=False)

    # 6. Image Assets Table
    op.create_table(
        "image_assets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("inspection_id", sa.Integer(), nullable=False),
        sa.Column("image_type", sa.String(length=32), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("camera_make", sa.String(length=128), nullable=True),
        sa.Column("camera_model", sa.String(length=128), nullable=True),
        sa.Column("emissivity", sa.Float(), nullable=True),
        sa.Column("reflected_temp_c", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["inspection_id"], ["inspections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_image_assets_checksum_sha256"), "image_assets", ["checksum_sha256"], unique=False)
    op.create_index(op.f("ix_image_assets_id"), "image_assets", ["id"], unique=False)
    op.create_index(op.f("ix_image_assets_image_type"), "image_assets", ["image_type"], unique=False)
    op.create_index(op.f("ix_image_assets_inspection_id"), "image_assets", ["inspection_id"], unique=False)


def downgrade() -> None:
    op.drop_table("image_assets")
    op.drop_table("inspections")
    op.drop_table("modules")
    op.drop_table("strings")
    op.drop_table("inverters")
    op.drop_table("plants")
