"""End-to-end integration tests for FastAPI transaction ingestion,

real-time GNN scoring, and motif alert detection.
"""

import pytest
from starlette.testclient import TestClient

from src.api.main import app, graph_store


@pytest.fixture(scope="module")
def client():
    """Test client with application lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /health returns engine status and node counts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "graph_node_count" in data
    assert "uptime_seconds" in data


def test_ingest_transaction_valid(client: TestClient):
    """Verify /ingest/transaction persists entities and updates counters."""
    payload = {
        "transaction_id": "tx_api_test_001",
        "source_account": "acc_api_alice",
        "target_account": "acc_api_bob",
        "amount": 250.75,
        "timestamp": 1716200000.0,
        "device_id": "dev_pixel_8",
        "ip_address": "198.51.100.12",
        "currency": "USD",
    }
    response = client.post("/ingest/transaction", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ingested"
    assert data["transaction_id"] == "tx_api_test_001"
    assert data["graph_updated"] is True
    assert data["current_nodes"] > 0


def test_ingest_transaction_invalid_amount(client: TestClient):
    """Verify input validation rejects non-positive monetary amounts."""
    payload = {
        "transaction_id": "tx_invalid_01",
        "source_account": "acc_alice",
        "target_account": "acc_bob",
        "amount": -50.0,  # Invalid amount
        "timestamp": 1716200000.0,
    }
    response = client.post("/ingest/transaction", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_score_transaction_legitimate_flow(client: TestClient):
    """Verify scoring a standard retail transaction yields low risk."""
    payload = {
        "transaction_id": "tx_score_retail_01",
        "source_account": "acc_normal_shopper",
        "target_account": "acc_local_bakery",
        "amount": 24.50,
        "timestamp": 1716200100.0,
        "device_id": "dev_retail_phone",
        "ip_address": "203.0.113.88",
    }
    # Initial warm-up request to prime JIT paths
    _ = client.post("/score/transaction", json=payload)

    response = client.post("/score/transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_score_retail_01"
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["inference_latency_ms"] > 0.0
    assert data["inference_latency_ms"] < 100.0  # Real-time latency bound
    assert data["is_flagged"] is False
    assert data["risk_level"] in ("LOW", "MEDIUM")


def test_score_transaction_injected_circular_mule_ring(client: TestClient):
    """Verify scoring an injected circular layering ring triggers high alert and motifs."""
    t0 = 1716250000.0
    # Ingest ring: Mule1 -> Mule2 -> Mule3 -> Mule1
    client.post("/ingest/transaction", json={
        "transaction_id": "tx_ring_h1", "source_account": "acc_circ_m1", "target_account": "acc_circ_m2",
        "amount": 12000.0, "timestamp": t0
    })
    client.post("/ingest/transaction", json={
        "transaction_id": "tx_ring_h2", "source_account": "acc_circ_m2", "target_account": "acc_circ_m3",
        "amount": 11400.0, "timestamp": t0 + 300.0
    })

    # Third hop completes the cycle: Mule3 -> Mule1
    closing_tx = {
        "transaction_id": "tx_ring_h3_closing",
        "source_account": "acc_circ_m3",
        "target_account": "acc_circ_m1",
        "amount": 10830.0,
        "timestamp": t0 + 600.0,
    }
    response = client.post("/score/transaction", json=closing_tx)
    assert response.status_code == 200
    data = response.json()

    assert data["fraud_score"] >= 0.65
    assert data["is_flagged"] is True
    assert data["risk_level"] in ("HIGH", "CRITICAL")

    # Verify explainable motif alert is present
    motifs = data["motifs"]
    assert len(motifs) >= 1
    cycle_motif = next((m for m in motifs if m["motif_type"] == "CIRCULAR_LAYERING"), None)
    assert cycle_motif is not None
    assert cycle_motif["severity"] == "CRITICAL"


def test_get_subgraph_endpoint(client: TestClient):
    """Verify /graph/subgraph/{account_id} returns serialized ego-subgraph."""
    response = client.get("/graph/subgraph/acc_circ_m1?k_hops=2")
    assert response.status_code == 200
    data = response.json()
    assert data["account_id"] == "acc_circ_m1"
    assert data["k_hops"] == 2
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 3
