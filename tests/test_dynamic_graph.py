"""Tests for dynamic heterogeneous graph store, k-hop ego-graph extraction,

and topological motif detection.
"""

import time
import pytest
import networkx as nx

from src.api.schemas import NodeType, RelationType, RiskLevel, TransactionEvent
from src.graph.dynamic_store import DynamicHeteroGraphStore
from src.graph.feature_extractor import GraphFeatureExtractor


@pytest.fixture
def store() -> DynamicHeteroGraphStore:
    """Fixture providing clean dynamic graph store instance."""
    s = DynamicHeteroGraphStore()
    s.clear()
    return s


@pytest.fixture
def extractor() -> GraphFeatureExtractor:
    """Fixture providing graph feature extractor instance."""
    return GraphFeatureExtractor()


def test_heterogeneous_node_and_edge_ingestion(store: DynamicHeteroGraphStore):
    """Verify Account, Device, and IP node creation with bi-directional relations."""
    event = TransactionEvent(
        transaction_id="tx_001",
        source_account="acc_alice",
        target_account="acc_bob",
        amount=500.0,
        timestamp=1716200000.0,
        device_id="dev_iphone_15",
        ip_address="192.168.1.50",
    )
    store.add_transaction(event)

    assert store.node_count == 4  # Alice, Bob, Device, IP
    assert store._graph.has_node("acc_alice")
    assert store._graph.has_node("acc_bob")
    assert store._graph.has_node("dev_iphone_15")
    assert store._graph.has_node("ip_192.168.1.50")

    # Verify node types
    assert store._graph.nodes["acc_alice"]["node_type"] == NodeType.ACCOUNT.value
    assert store._graph.nodes["dev_iphone_15"]["node_type"] == NodeType.DEVICE.value
    assert store._graph.nodes["ip_192.168.1.50"]["node_type"] == NodeType.IP.value

    # Verify edge relations
    assert store._graph["acc_alice"]["acc_bob"]["relation"] == RelationType.TRANSFERS_TO.value
    assert store._graph["acc_alice"]["dev_iphone_15"]["relation"] == RelationType.USED_DEVICE.value
    assert store._graph["dev_iphone_15"]["acc_alice"]["relation"] == RelationType.DEVICE_OF.value
    assert store._graph["acc_alice"]["ip_192.168.1.50"]["relation"] == RelationType.ACCESSED_FROM_IP.value
    assert store._graph["ip_192.168.1.50"]["acc_alice"]["relation"] == RelationType.IP_OF.value


def test_k_hop_ego_graph_extraction(store: DynamicHeteroGraphStore):
    """Verify 1-hop and 2-hop bounded ego-graph extraction."""
    # Build a 3-hop chain: Alice -> Bob -> Charlie -> Dave
    store.add_transaction(TransactionEvent(
        transaction_id="tx_1", source_account="Alice", target_account="Bob",
        amount=100.0, timestamp=1000.0, device_id="dev_1"
    ))
    store.add_transaction(TransactionEvent(
        transaction_id="tx_2", source_account="Bob", target_account="Charlie",
        amount=100.0, timestamp=1010.0, device_id="dev_2"
    ))
    store.add_transaction(TransactionEvent(
        transaction_id="tx_3", source_account="Charlie", target_account="Dave",
        amount=100.0, timestamp=1020.0, device_id="dev_3"
    ))

    # 1-hop around Bob includes Alice, Bob, Charlie, dev_1, dev_2
    ego_1 = store.get_k_hop_subgraph("Bob", k=1)
    assert "Alice" in ego_1.nodes
    assert "Bob" in ego_1.nodes
    assert "Charlie" in ego_1.nodes
    assert "Dave" not in ego_1.nodes  # Dave is 2 hops away

    # 2-hop around Bob includes Dave
    ego_2 = store.get_k_hop_subgraph("Bob", k=2)
    assert "Dave" in ego_2.nodes


def test_circular_layering_motif_detection(store: DynamicHeteroGraphStore):
    """Verify money mule circular layering detection: A -> B -> C -> A."""
    t0 = 1716200000.0
    store.add_transaction(TransactionEvent(
        transaction_id="tx_c1", source_account="Mule_A", target_account="Mule_B",
        amount=10000.0, timestamp=t0
    ))
    store.add_transaction(TransactionEvent(
        transaction_id="tx_c2", source_account="Mule_B", target_account="Mule_C",
        amount=9500.0, timestamp=t0 + 600.0
    ))
    store.add_transaction(TransactionEvent(
        transaction_id="tx_c3", source_account="Mule_C", target_account="Mule_A",
        amount=9000.0, timestamp=t0 + 1200.0
    ))

    motifs = store.detect_motifs("Mule_A")
    assert len(motifs) >= 1
    cycle_motif = next((m for m in motifs if m.motif_type == "CIRCULAR_LAYERING"), None)
    assert cycle_motif is not None
    assert cycle_motif.severity == RiskLevel.CRITICAL
    assert set(cycle_motif.involved_nodes) == {"Mule_A", "Mule_B", "Mule_C"}


def test_device_collusion_ring_motif_detection(store: DynamicHeteroGraphStore):
    """Verify shared emulator device collusion detection across >= 3 accounts."""
    shared_dev = "dev_emulator_syndicate"
    t0 = 1716200000.0

    for i in range(4):
        store.add_transaction(TransactionEvent(
            transaction_id=f"tx_dev_{i}",
            source_account=f"Syndicate_Acc_{i}",
            target_account="Cashout_Drop",
            amount=5000.0,
            timestamp=t0 + (i * 100),
            device_id=shared_dev,
        ))

    motifs = store.detect_motifs("Syndicate_Acc_0")
    dev_motif = next((m for m in motifs if m.motif_type == "DEVICE_COLLUSION_RING"), None)
    assert dev_motif is not None
    assert dev_motif.severity == RiskLevel.CRITICAL
    assert f"dev_{shared_dev}" in dev_motif.involved_nodes or shared_dev in dev_motif.involved_nodes


def test_smurfing_fan_out_detection(store: DynamicHeteroGraphStore):
    """Verify smurfing fan-out structuring detection from kingpin account."""
    t0 = 1716200000.0
    kingpin = "acc_kingpin_test"

    for i in range(6):
        store.add_transaction(TransactionEvent(
            transaction_id=f"tx_fo_{i}",
            source_account=kingpin,
            target_account=f"acc_smurf_dest_{i}",
            amount=9200.0,
            timestamp=t0 + (i * 60),
        ))

    motifs = store.detect_motifs(kingpin)
    smurf_motif = next((m for m in motifs if m.motif_type == "SMURFING_FAN_OUT"), None)
    assert smurf_motif is not None
    assert smurf_motif.severity == RiskLevel.HIGH


def test_feature_extractor_structural_signals(store: DynamicHeteroGraphStore, extractor: GraphFeatureExtractor):
    """Verify structural graph features are correctly computed."""
    event = TransactionEvent(
        transaction_id="tx_feat_01",
        source_account="Acc_Sender",
        target_account="Acc_Receiver",
        amount=1500.0,
        timestamp=1716200000.0,
        device_id="dev_phone_1",
    )
    store.add_transaction(event)

    ego = store.get_k_hop_subgraph("Acc_Sender", k=2)
    signals = extractor.extract_structural_signals(event, ego)

    for feat in GraphFeatureExtractor.FEATURE_NAMES:
        assert feat in signals, f"Missing feature {feat}"
        assert isinstance(signals[feat], float)
        assert not (signals[feat] != signals[feat])  # Check not NaN
