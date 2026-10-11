"""Contextual Diagnostic Assistant dialogue endpoints."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.knowledge.assistant import ContextualAssistant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Diagnostic Chatbot"])


class ChatQueryRequest(BaseModel):
    """Dialogue turn request schema."""

    query: str = Field(..., description="User question for the diagnostic assistant")
    session_id: Optional[str] = Field(None, description="Persistent conversation session ID")
    plant_id: Optional[int] = Field(None, description="Target solar plant ID")
    inspection_id: Optional[int] = Field(None, description="Target inspection ID for grounding")


class ChatQueryResponse(BaseModel):
    """Dialogue turn response schema."""

    session_id: str
    response: str
    citations: List[str]
    has_inspection_context: bool
    has_diagnosis_context: bool


@router.post("/message", response_model=ChatQueryResponse)
def post_chat_message(
    req: ChatQueryRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Consult the contextual diagnostic assistant with multi-modal inspection grounding and standards RAG."""
    if not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    try:
        assistant = ContextualAssistant(db)
        return assistant.answer_query(
            query=req.query,
            session_id=req.session_id,
            plant_id=req.plant_id,
            inspection_id=req.inspection_id,
        )
    except Exception as exc:
        logger.exception("Failed to process chat query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assistant computation error: {str(exc)}",
        )


@router.get("/sessions/{session_id}/history")
def get_session_history(
    session_id: str,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Retrieve full conversation history for a chat session."""
    try:
        assistant = ContextualAssistant(db)
        return assistant.get_session_history(session_id)
    except Exception as exc:
        logger.exception("Failed to retrieve chat history for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(exc)}",
        )

