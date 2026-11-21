"""Tests for PyTorch Relational Graph Convolutional Network (R-GCN) forward pass,

deterministic seeding, weight persistence, and sub-15ms CPU inference latency.
"""

import os
import tempfile
import time
import networkx as nx
import numpy as np
import pytest
import torch

from src.api.schemas import RelationType
from src.graph.feature_extractor import GraphFeatureExtractor
from src.models.rgcn_model import NUM_RELATIONS, RELATION_MAP, RGCNFraudDetector, RelationalGraphConvLayer


def test_rgcn_layer_message_passing():
    """Verify single RelationalGraphConvLayer forward pass and output shape."""
    N = 10
    in_dim = 16
    out_dim = 32

    layer = RelationalGraphConvLayer(in_features=in_dim, out_features=out_dim)
    x = torch.randn(N, in_dim)

    # Edge dict with relations
    edge_dict = {
        0: torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        2: torch.tensor([[0, 4], [4, 5]], dtype=torch.long),
    }

    out = layer(x, edge_dict)
    assert out.shape == (N, out_dim)
    assert not torch.isnan(out).any()


def test_rgcn_model_forward_and_scoring():
    """Verify RGCNFraudDetector end-to-end forward pass and scoring bounds."""
    model = RGCNFraudDetector(
        node_in_dim=16,
        hidden_dim=32,
        edge_feature_dim=8,
        num_layers=2,
        seed=42,
    )
    model.eval()

    # Create dummy ego-graph with 5 nodes
    graph = nx.DiGraph()
    graph.add_node("acc_1", node_type="ACCOUNT")
    graph.add_node("acc_2", node_type="ACCOUNT")
    graph.add_node("dev_1", node_type="DEVICE")
    graph.add_edge("acc_1", "acc_2", relation=RelationType.TRANSFERS_TO.value)
    graph.add_edge("acc_1", "dev_1", relation=RelationType.USED_DEVICE.value)

    extractor = GraphFeatureExtractor()
    node_order = list(graph.nodes)
    node_feats = extractor.extract_node_features_matrix(graph, node_order)
    edge_feats = np.zeros(8, dtype=np.float32)

    score = model.score_subgraph(
        graph=graph,
        source_id="acc_1",
        target_id="acc_2",
        node_features_matrix=node_feats,
        node_order=node_order,
        edge_features=edge_feats,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_model_weight_determinism():
    """Verify deterministic initialization yields bit-for-bit identical predictions."""
    model1 = RGCNFraudDetector(seed=123)
    model1.eval()
    model2 = RGCNFraudDetector(seed=123)
    model2.eval()

    torch.manual_seed(999)
    x = torch.randn(5, 16)
    edges = {0: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)}

    with torch.no_grad():
        h1 = model1.forward_embeddings(x, edges)
        h2 = model2.forward_embeddings(x, edges)

    assert torch.allclose(h1, h2, atol=1e-6)


def test_save_and_load_pretrained():
    """Verify model persistence to disk and round-trip fidelity."""
    model = RGCNFraudDetector(node_in_dim=16, hidden_dim=32, seed=42)
    model.eval()
    torch.manual_seed(999)
    x = torch.randn(4, 16)
    edges = {0: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)}

    with torch.no_grad():
        orig_out = model.forward_embeddings(x, edges)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "rgcn_test.pt")
        model.save_pretrained(ckpt_path)
        loaded_model = RGCNFraudDetector.load_pretrained(ckpt_path)

        with torch.no_grad():
            loaded_out = loaded_model.forward_embeddings(x, edges)

        assert torch.allclose(orig_out, loaded_out, atol=1e-6)


def test_sub_15ms_inference_latency_on_cpu():
    """Verify that forward inference on a realistic ego-graph completes within 15ms on CPU."""
    model = RGCNFraudDetector(
        node_in_dim=16,
        hidden_dim=64,
        edge_feature_dim=8,
        num_layers=2,
        seed=42,
    )
    model.eval()

    # Construct realistic 50-node ego-graph
    graph = nx.DiGraph()
    for i in range(40):
        graph.add_node(f"acc_{i}", node_type="ACCOUNT")
    for d in range(10):
        graph.add_node(f"dev_{d}", node_type="DEVICE")

    for i in range(39):
        graph.add_edge(f"acc_{i}", f"acc_{i+1}", relation=RelationType.TRANSFERS_TO.value)
        graph.add_edge(f"acc_{i}", f"dev_{i % 10}", relation=RelationType.USED_DEVICE.value)

    extractor = GraphFeatureExtractor()
    node_order = list(graph.nodes)
    node_feats = extractor.extract_node_features_matrix(graph, node_order)
    edge_feats = np.zeros(8, dtype=np.float32)

    # Warm-up pass
    for _ in range(5):
        _ = model.score_subgraph(
            graph=graph,
            source_id="acc_0",
            target_id="acc_1",
            node_features_matrix=node_feats,
            node_order=node_order,
            edge_features=edge_feats,
        )

    latencies_ms = []
    num_runs = 50
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.score_subgraph(
            graph=graph,
            source_id="acc_0",
            target_id="acc_1",
            node_features_matrix=node_feats,
            node_order=node_order,
            edge_features=edge_feats,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_latency = float(np.percentile(latencies_ms, 50))
    p95_latency = float(np.percentile(latencies_ms, 95))

    print(f"\n[LATENCY BENCHMARK] P50: {p50_latency:.2f} ms | P95: {p95_latency:.2f} ms")
    assert p50_latency < 15.0, f"P50 latency {p50_latency:.2f}ms exceeded 15ms target!"
