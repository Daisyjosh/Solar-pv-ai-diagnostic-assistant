"""Database ORM models for the contextual diagnostic assistant and chat history."""

from datetime import datetime, timezone
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.session import Base

if TYPE_CHECKING:
    from src.database.models.inspection import Inspection
    from src.database.models.plant import Plant


class ChatSession(Base):
    """Context-aware chat session linked to a solar plant or inspection event."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    plant_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=True, index=True
    )
    inspection_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=True, index=True
    )
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

    plant: Mapped[Optional["Plant"]] = relationship("Plant")
    inspection: Mapped[Optional["Inspection"]] = relationship("Inspection")
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at.asc()",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "title": self.title,
            "plant_id": self.plant_id,
            "inspection_id": self.inspection_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
        }


class ChatMessage(Base):
    """Individual dialogue turn in an assistant conversation."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "citations": json.loads(self.citations_json),
            "created_at": self.created_at.isoformat(),
        }
