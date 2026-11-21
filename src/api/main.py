"""Production-grade FastAPI application for real-time streaming transaction ingestion,

heterogeneous graph ego-subgraph extraction, GNN fraud scoring, and explainable motif alerts.
"""

from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    FraudScoreResponse,
    HealthResponse,
    IngestResponse,
    RiskLevel,
    SubgraphResponse,
    TransactionEvent,
)
from src.graph.dynamic_store import DynamicHeteroGraphStore
from src.graph.feature_extractor import GraphFeatureExtractor
from src.models.rgcn_model import RGCNFraudDetector
from src.models.tabular_baseline import TabularFraudBaseline
from src.simulation.fraud_ring_generator import FinancialNetworkSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("FraudGraph-API")

# Global engine singletons
graph_store = DynamicHeteroGraphStore()
feature_extractor = GraphFeatureExtractor()
rgcn_model = RGCNFraudDetector(
    node_in_dim=16,
    hidden_dim=64,
    edge_feature_dim=8,
    num_layers=2,
    seed=42,
)
tabular_baseline = TabularFraudBaseline(random_state=42)
SIMULATION_MODE: bool = os.getenv("SIMULATION_MODE", "true").lower() in ("true", "1", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize models and warm up graph cache."""
    logger.info("Initializing FraudGraph-GNN engine components...")

    if SIMULATION_MODE:
        logger.info("SIMULATION_MODE=true: Pre-populating dynamic store with warm-up financial network...")
        warmup_sim = FinancialNetworkSimulator(num_transactions=1200, num_rings=12, seed=42)
        warmup_txs = warmup_sim.generate_dataset()

        # Ingest into store
        for tx in warmup_txs:
            graph_store.add_transaction(tx)

        # Fit tabular baseline on warm-up data
        X_tab = [feature_extractor.extract_tabular_features(tx) for tx in warmup_txs]
        y_tab = [1 if tx.is_fraud else 0 for tx in warmup_txs]
        import numpy as np

        tabular_baseline.fit(np.array(X_tab), np.array(y_tab))
        logger.info(
            f"Warm-up complete: {graph_store.node_count} nodes, "
            f"{graph_store.edge_count} edges loaded into graph store."
        )

    yield

    logger.info("Shutting down FraudGraph-GNN engine...")


app = FastAPI(
    title="FraudGraph-GNN Engine",
    description="Real-time Financial Fraud and Money-Mule Ring Detection Engine powered by PyTorch R-GCN & NetworkX",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Service health, graph node/edge counts, and uptime metrics."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        graph_node_count=graph_store.node_count,
        graph_edge_count=graph_store.edge_count,
        uptime_seconds=round(graph_store.uptime_seconds, 2),
        simulation_mode=SIMULATION_MODE,
    )


@app.post(
    "/ingest/transaction",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def ingest_transaction(event: TransactionEvent) -> IngestResponse:
    """Stream ingest a transaction event into the dynamic heterogeneous graph store."""
    try:
        graph_store.add_transaction(event)
        return IngestResponse(
            status="ingested",
            transaction_id=event.transaction_id,
            graph_updated=True,
            current_nodes=graph_store.node_count,
            current_edges=graph_store.edge_count,
        )
    except Exception as e:
        logger.error(f"Error ingesting transaction {event.transaction_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest transaction: {str(e)}",
        )


@app.post(
    "/score/transaction",
    response_model=FraudScoreResponse,
    tags=["Fraud Scoring"],
)
async def score_transaction(event: TransactionEvent) -> FraudScoreResponse:
    """Ingest and score a transaction in real-time (<15ms) using PyTorch R-GCN message passing,

    extracting topological features and explainable fraud motif alerts.
    """
    start_time = time.perf_counter()

    # 1. Update graph with incoming event
    graph_store.add_transaction(event)

    # 2. Extract k-hop ego-graph around source and destination
    ego_src = graph_store.get_k_hop_subgraph(event.source_account, k=2, max_nodes=100)
    ego_dst = graph_store.get_k_hop_subgraph(event.target_account, k=1, max_nodes=50)

    # Union ego-graphs for complete interaction context
    import networkx as nx
    combined_subgraph = nx.compose(ego_src, ego_dst)

    # 3. Detect explainable topological motifs
    motifs_src = graph_store.detect_motifs(event.source_account, combined_subgraph)
    motifs_dst = graph_store.detect_motifs(event.target_account, combined_subgraph)

    # Deduplicate motifs
    seen_motifs = set()
    motifs = []
    for m in motifs_src + motifs_dst:
        key = (m.motif_type, tuple(sorted(m.involved_nodes)))
        if key not in seen_motifs:
            seen_motifs.add(key)
            motifs.append(m)

    # 4. Extract structural signals
    signals = feature_extractor.extract_structural_signals(event, combined_subgraph)

    # 5. Prepare tensors for R-GCN forward inference
    node_order = list(combined_subgraph.nodes)
    node_features_matrix = feature_extractor.extract_node_features_matrix(
        combined_subgraph, node_order
    )
    edge_features = feature_extractor.extract_tabular_features(event, combined_subgraph)

    # 6. GNN scoring
    raw_gnn_score = rgcn_model.score_subgraph(
        graph=combined_subgraph,
        source_id=event.source_account,
        target_id=event.target_account,
        node_features_matrix=node_features_matrix,
        node_order=node_order,
        edge_features=edge_features,
    )

    # 7. Motif-informed calibration
    # If a verified topological money mule cycle or collusion ring is present, elevate confidence
    final_score = raw_gnn_score
    if motifs:
        max_motif_conf = max(m.confidence for m in motifs)
        has_critical = any(m.severity == RiskLevel.CRITICAL for m in motifs)
        if has_critical:
            final_score = max(final_score, max_motif_conf * 0.94)
        else:
            final_score = max(final_score, max_motif_conf * 0.78)

    final_score = float(max(0.01, min(0.99, final_score)))

    # Determine risk category & alert trigger
    if final_score >= 0.85:
        risk = RiskLevel.CRITICAL
        flagged = True
    elif final_score >= 0.65:
        risk = RiskLevel.HIGH
        flagged = True
    elif final_score >= 0.40:
        risk = RiskLevel.MEDIUM
        flagged = False
    else:
        risk = RiskLevel.LOW
        flagged = False

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return FraudScoreResponse(
        transaction_id=event.transaction_id,
        source_account=event.source_account,
        target_account=event.target_account,
        fraud_score=round(final_score, 4),
        is_flagged=flagged,
        risk_level=risk,
        inference_latency_ms=round(latency_ms, 2),
        structural_signals=signals,
        motifs=motifs,
        model_version="FraudGraph-RGCN-v1.0.0",
    )


@app.get(
    "/graph/subgraph/{account_id}",
    response_model=SubgraphResponse,
    tags=["Graph Inspection"],
)
async def get_account_subgraph(
    account_id: str,
    k_hops: int = Query(2, ge=1, le=4, description="Ego-graph hop radius"),
) -> SubgraphResponse:
    """Retrieve the k-hop ego-graph centered on an account for forensic inspection."""
    return graph_store.export_subgraph_response(account_id=account_id, k_hops=k_hops)
