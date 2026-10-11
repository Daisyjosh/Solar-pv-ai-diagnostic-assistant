"""Telemetry Ingestion Engine supporting CSV and Excel formats.

Performs robust column mapping, unit normalization, timestamp parsing,
data-quality validation, and transaction-safe database ingestion.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.inverter import Inverter
from src.database.models.string import PVString
from src.database.models.telemetry import TelemetryBatch, TelemetryRecord

logger = logging.getLogger(__name__)

# Standard flexible header mapping dictionaries
COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "ts", "utc_time", "reading_time", "date"],
    "voltage_v": ["voltage", "voltage_v", "v_dc", "vdc", "v_mppt", "volts", "v", "dc_voltage"],
    "current_a": ["current", "current_a", "i_dc", "idc", "i_string", "amps", "a", "dc_current"],
    "power_kw": ["power", "power_kw", "power_w", "p_dc", "p_ac", "kw", "w", "active_power"],
    "irradiance_w_m2": ["irradiance", "irradiance_w_m2", "poa", "ghi", "solar_rad", "radiation", "flux"],
    "ambient_temp_c": ["ambient_temp", "ambient_temp_c", "t_amb", "temp_ambient", "ambient", "ambient_c"],
    "module_temp_c": ["module_temp", "module_temp_c", "t_mod", "temp_module", "cell_temp", "module_c"],
    "operating_status": ["status", "operating_status", "state", "inverter_status", "alarm_status"],
    "inverter_code": ["inverter_code", "inverter", "inv_code", "inv", "inverter_id"],
    "string_code": ["string_code", "string", "str_code", "str", "string_id"],
}


@dataclass
class ColumnMappingResult:
    """Detected column mapping from uploaded file headers."""

    mapped_columns: Dict[str, str]  # standard_field -> raw_file_header
    unmapped_headers: List[str]
    missing_required_fields: List[str]


@dataclass
class TelemetryPreview:
    """Preview report returned to user prior to committing import."""

    filename: str
    total_rows: int
    valid_rows_count: int
    column_mapping: Dict[str, str]
    preview_records: List[Dict[str, Any]]
    validation_warnings: List[str]
    detected_time_range: Dict[str, Optional[str]]


class TelemetryImporter:
    """Service handling multi-format telemetry validation and ingestion."""

    def __init__(self, db_session: Session):
        self.db = db_session

    @staticmethod
    def _detect_columns(raw_headers: List[str]) -> ColumnMappingResult:
        """Fuzzy match raw table headers to standard electrical fields."""
        norm_headers = {h.strip().lower().replace(" ", "_").replace("-", "_"): h for h in raw_headers}
        mapped: Dict[str, str] = {}

        for std_field, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in norm_headers:
                    mapped[std_field] = norm_headers[alias]
                    break

        unmapped = [h for h in raw_headers if h not in mapped.values()]
        missing_req = []
        if "timestamp" not in mapped:
            missing_req.append("timestamp")

        return ColumnMappingResult(
            mapped_columns=mapped,
            unmapped_headers=unmapped,
            missing_required_fields=missing_req,
        )

    @classmethod
    def parse_dataframe(cls, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """Parse raw file bytes into pandas DataFrame based on file extension."""
        suffix = Path(filename).suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        elif suffix in (".csv", ".txt"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .csv, .xlsx, .xls")

    def preview_import(self, file_bytes: bytes, filename: str) -> TelemetryPreview:
        """Generate validation preview and column mapping without committing."""
        df = self.parse_dataframe(file_bytes, filename)
        if df.empty:
            raise ValueError("Uploaded file contains no rows.")

        col_result = self._detect_columns(list(df.columns))
        if col_result.missing_required_fields:
            raise ValueError(
                f"Missing mandatory column: {col_result.missing_required_fields}. "
                f"File headers found: {list(df.columns)}"
            )

        ts_col = col_result.mapped_columns["timestamp"]
        warnings = []

        # Parse timestamps safely
        try:
            df["_parsed_ts"] = pd.to_datetime(df[ts_col], utc=True)
        except Exception as exc:
            raise ValueError(f"Failed to parse timestamp column '{ts_col}': {str(exc)}")

        # Check for null timestamps
        null_ts = df["_parsed_ts"].isna().sum()
        if null_ts > 0:
            warnings.append(f"{null_ts} rows have missing/unparseable timestamps and will be omitted.")

        valid_df = df.dropna(subset=["_parsed_ts"]).copy()
        if valid_df.empty:
            raise ValueError("All rows contain invalid or empty timestamps.")

        min_time = valid_df["_parsed_ts"].min().isoformat()
        max_time = valid_df["_parsed_ts"].max().isoformat()

        # Build preview sample (first 10 valid rows)
        preview_rows = []
        for idx, row in valid_df.head(10).iterrows():
            rec = {"row_number": int(idx) + 1, "timestamp": row["_parsed_ts"].isoformat()}
            for std_field, raw_col in col_result.mapped_columns.items():
                if std_field == "timestamp":
                    continue
                val = row.get(raw_col)
                if pd.notna(val):
                    rec[std_field] = round(float(val), 3) if isinstance(val, (int, float, np.number)) else str(val)
                else:
                    rec[std_field] = None
            preview_rows.append(rec)

        return TelemetryPreview(
            filename=filename,
            total_rows=len(df),
            valid_rows_count=len(valid_df),
            column_mapping=col_result.mapped_columns,
            preview_records=preview_rows,
            validation_warnings=warnings,
            detected_time_range={"start_time": min_time, "end_time": max_time},
        )

    def commit_import(
        self,
        plant_id: int,
        file_bytes: bytes,
        filename: str,
        notes: Optional[str] = None,
    ) -> TelemetryBatch:
        """Ingest and commit telemetry records into the database."""
        df = self.parse_dataframe(file_bytes, filename)
        col_result = self._detect_columns(list(df.columns))

        if col_result.missing_required_fields:
            raise ValueError(f"Missing mandatory column: {col_result.missing_required_fields}")

        ts_col = col_result.mapped_columns["timestamp"]
        df["_parsed_ts"] = pd.to_datetime(df[ts_col], utc=True)
        valid_df = df.dropna(subset=["_parsed_ts"]).copy()

        if valid_df.empty:
            raise ValueError("No valid timestamped records found to import.")

        # Cache plant's inverters and strings for rapid foreign-key association
        inverters = self.db.scalars(select(Inverter).where(Inverter.plant_id == plant_id)).all()
        inv_map = {inv.code.strip().lower(): inv.id for inv in inverters}

        inv_ids = [inv.id for inv in inverters]
        strings = self.db.scalars(select(PVString).where(PVString.inverter_id.in_(inv_ids))).all() if inv_ids else []
        str_map = {s.code.strip().lower(): s.id for s in strings}

        file_type = "EXCEL" if Path(filename).suffix.lower() in (".xlsx", ".xls") else "CSV"
        min_time = valid_df["_parsed_ts"].min().to_pydatetime()
        max_time = valid_df["_parsed_ts"].max().to_pydatetime()

        # Create TelemetryBatch
        batch = TelemetryBatch(
            plant_id=plant_id,
            filename=filename,
            file_type=file_type,
            records_count=len(valid_df),
            start_time=min_time,
            end_time=max_time,
            status="COMMITTED",
            import_notes=notes,
        )
        self.db.add(batch)
        self.db.flush()

        # Ingest records in chunks
        records_to_insert = []
        for _, row in valid_df.iterrows():
            ts = row["_parsed_ts"].to_pydatetime()

            # Helper to extract and sanitize float
            def get_float(field: str) -> Optional[float]:
                raw_col = col_result.mapped_columns.get(field)
                if not raw_col or pd.isna(row.get(raw_col)):
                    return None
                try:
                    val = float(row[raw_col])
                    # Auto-convert watts to kW if value > 2000 W and field is power
                    if field == "power_kw" and val > 2000.0:
                        val = val / 1000.0
                    return val
                except (ValueError, TypeError):
                    return None

            # Equipment resolution
            inv_id = None
            if "inverter_code" in col_result.mapped_columns:
                code_raw = str(row.get(col_result.mapped_columns["inverter_code"], "")).strip().lower()
                inv_id = inv_map.get(code_raw)

            string_id = None
            if "string_code" in col_result.mapped_columns:
                code_raw = str(row.get(col_result.mapped_columns["string_code"], "")).strip().lower()
                string_id = str_map.get(code_raw)

            status_val = None
            if "operating_status" in col_result.mapped_columns:
                val = row.get(col_result.mapped_columns["operating_status"])
                if pd.notna(val):
                    status_val = str(val).strip()

            rec = TelemetryRecord(
                batch_id=batch.id,
                plant_id=plant_id,
                inverter_id=inv_id,
                string_id=string_id,
                timestamp=ts,
                voltage_v=get_float("voltage_v"),
                current_a=get_float("current_a"),
                power_kw=get_float("power_kw"),
                irradiance_w_m2=get_float("irradiance_w_m2"),
                ambient_temp_c=get_float("ambient_temp_c"),
                module_temp_c=get_float("module_temp_c"),
                operating_status=status_val,
            )
            records_to_insert.append(rec)

        self.db.add_all(records_to_insert)
        self.db.commit()
        self.db.refresh(batch)
        logger.info("Successfully committed telemetry batch %d (%d records)", batch.id, len(records_to_insert))
        return batch
