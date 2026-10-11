"""Unit tests for Phase 8 Diagnostic Assistant and RAG Document Store."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models.inspection import Inspection
from src.database.models.plant import Plant
from src.database.session import Base
from src.knowledge.assistant import ContextualAssistant
from src.knowledge.document_store import KnowledgeDocumentStore


def test_knowledge_document_store_retrieval():
    store = KnowledgeDocumentStore()
    results = store.search("minimum irradiance for thermal thermography inspection", top_k=2)

    assert len(results) >= 1
    top_hit = results[0]
    assert "IEC 62446-3" in top_hit["standard_reference"]
    assert "600 W/m2" in top_hit["content"]
    assert top_hit["similarity_score"] > 0.1


def test_contextual_assistant_response_with_inspection_context():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)
    session = session_local()

    plant = Plant(code="P-TEST", name="Sahara Solar", location="Morocco", capacity_dc_mw=100.0)
    session.add(plant)
    session.commit()

    insp = Inspection(
        code="INSP-TEST-01",
        plant_id=plant.id,
        inspector_name="Lead Inspector",
        ambient_temp_c=30.0,
        irradiance_w_m2=820.0,
    )
    session.add(insp)
    session.commit()

    assistant = ContextualAssistant(session)
    res = assistant.answer_query(
        query="What are the environmental conditions and safety precautions for this inspection?",
        plant_id=plant.id,
        inspection_id=insp.id,
    )

    assert res["has_inspection_context"] is True
    assert "Sahara Solar" in res["response"]
    assert "820" in res["response"]
    assert len(res["citations"]) >= 1

    # Verify message persistence
    history = assistant.get_session_history(res["session_id"])
    assert len(history) == 2  # user query + assistant response
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_contextual_assistant_with_diagnosis_and_confidence_formatting():
    """Regression test: verify assistant handles diagnosis records, confidence formatting, and hypothesis status."""
    import json
    from src.database.models.diagnosis import Diagnosis

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)
    session = session_local()

    plant = Plant(code="P-TEST-DIAG", name="Apex Solar", location="Spain", capacity_dc_mw=50.0)
    session.add(plant)
    session.commit()

    insp = Inspection(
        code="INSP-TEST-DIAG",
        plant_id=plant.id,
        inspector_name="Lead Engineer",
        ambient_temp_c=28.0,
        irradiance_w_m2=750.0,
    )
    session.add(insp)
    session.commit()

    # Add diagnosis record with float model_confidence and INCONCLUSIVE severity
    diag = Diagnosis(
        inspection_id=insp.id,
        assessment_id="DIAG-TEST-01",
        fault_category="Suspected Cell Microcrack (Uncorroborated AI Hypothesis)",
        canonical_fault_code="FLT-CELL-003",
        severity="INCONCLUSIVE",
        severity_justification="AI hypothesis requires physical corroboration.",
        model_prediction="Cracking",
        model_confidence=0.311,
        power_loss_percent=None,
        power_loss_justification="Cannot be quantified without I-V trace.",
        root_causes_json=json.dumps(["Hypothesis A: Microcrack", "Hypothesis B: Cell-Multi"]),
        corrective_actions_json=json.dumps(["Perform EL imaging"]),
        preventive_actions_json=json.dumps(["Avoid walking on panels"]),
        safety_precautions_json=json.dumps(["Use insulated tools"]),
        required_measurements_json=json.dumps(["Dark EL imaging", "Module I-V curve"]),
        diagnostic_limitations_json=json.dumps(["Surface thermography only"]),
        raw_assessment_json="{}",
        review_status="PENDING_REVIEW",
    )
    session.add(diag)
    session.commit()

    assistant = ContextualAssistant(session)
    # Must not raise ValueError: Invalid format specifier
    res = assistant.answer_query(
        query="What is the fault and required measurements for this inspection?",
        inspection_id=insp.id,
    )

    assert res["has_inspection_context"] is True
    assert res["has_diagnosis_context"] is True
    resp_text = res["response"]
    assert "31.1%" in resp_text
    assert "INCONCLUSIVE" in resp_text
    assert "Unquantified" in resp_text
    assert "Uncorroborated AI Hypothesis" in resp_text
    assert "Engineer Review Pending" in resp_text
    assert "No stored string/inverter telemetry records exist" in resp_text
    assert any("IEC 62446-3" in c for c in res["citations"])


def test_api_client_send_chat_message_resilience(monkeypatch):
    """Regression test: verify send_chat_message safely handles plaintext and non-JSON error responses."""
    from frontend.api_client import BackendAPIClient

    client = BackendAPIClient(base_url="http://mock-server")

    class MockResponse:
        status_code = 500
        text = "Internal Server Error"
        def json(self):
            import json
            raise json.decoder.JSONDecodeError("Expecting value", "Internal Server Error", 0)

    class MockHttpxClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json=None):
            return MockResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", MockHttpxClient)

    # Must NOT raise JSONDecodeError or crash with char 0 error
    res = client.send_chat_message(query="Test query")
    assert "response" in res
    assert "Assistant error (500): Internal Server Error" in res["response"]

