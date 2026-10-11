"""Telemetry analytics and historical trend retrieval service."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models.telemetry import TelemetryBatch, TelemetryRecord


class TelemetryAnalyticsService:
    """Service providing aggregated measurements and time-series trends."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_batches(self, plant_id: int) -> List[Dict[str, Any]]:
        """List all committed telemetry batches for a plant."""
        stmt = (
            select(TelemetryBatch)
            .where(TelemetryBatch.plant_id == plant_id)
            .order_by(TelemetryBatch.created_at.desc())
        )
        batches = self.db.scalars(stmt).all()
        return [b.to_dict() for b in batches]

    def get_records(
        self,
        plant_id: int,
        inverter_id: Optional[int] = None,
        string_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query time-series records filtered by equipment and time window."""
        stmt = select(TelemetryRecord).where(TelemetryRecord.plant_id == plant_id)
        if inverter_id is not None:
            stmt = stmt.where(TelemetryRecord.inverter_id == inverter_id)
        if string_id is not None:
            stmt = stmt.where(TelemetryRecord.string_id == string_id)
        if start_time is not None:
            stmt = stmt.where(TelemetryRecord.timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(TelemetryRecord.timestamp <= end_time)

        stmt = stmt.order_by(TelemetryRecord.timestamp.asc()).limit(limit)
        records = self.db.scalars(stmt).all()
        return [r.to_dict() for r in records]

    def get_trend_summary(self, plant_id: int, hours: int = 24) -> Dict[str, Any]:
        """Compute aggregated statistics and time-series curve for dashboard charts."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        stmt = (
            select(TelemetryRecord)
            .where(TelemetryRecord.plant_id == plant_id)
            .where(TelemetryRecord.timestamp >= cutoff)
            .order_by(TelemetryRecord.timestamp.asc())
        )
        records = self.db.scalars(stmt).all()

        if not records:
            # Fall back to latest records regardless of time window if recent data not present
            stmt_latest = (
                select(TelemetryRecord)
                .where(TelemetryRecord.plant_id == plant_id)
                .order_by(TelemetryRecord.timestamp.desc())
                .limit(200)
            )
            records = list(reversed(self.db.scalars(stmt_latest).all()))

        if not records:
            return {
                "has_data": False,
                "message": "No telemetry records found for this plant.",
                "record_count": 0,
                "series": {},
                "summary": {},
            }

        data_rows = [
            {
                "timestamp": r.timestamp.isoformat(),
                "voltage_v": r.voltage_v,
                "current_a": r.current_a,
                "power_kw": r.power_kw,
                "irradiance_w_m2": r.irradiance_w_m2,
                "ambient_temp_c": r.ambient_temp_c,
                "module_temp_c": r.module_temp_c,
            }
            for r in records
        ]
        df = pd.DataFrame(data_rows)

        summary = {}
        for col in ["voltage_v", "current_a", "power_kw", "irradiance_w_m2", "module_temp_c"]:
            valid = df[col].dropna()
            if not valid.empty:
                summary[col] = {
                    "mean": round(float(valid.mean()), 2),
                    "max": round(float(valid.max()), 2),
                    "min": round(float(valid.min()), 2),
                    "latest": round(float(valid.iloc[-1]), 2),
                }

        return {
            "has_data": True,
            "record_count": len(records),
            "start_time": df["timestamp"].iloc[0],
            "end_time": df["timestamp"].iloc[-1],
            "series": data_rows,
            "summary": summary,
        }
