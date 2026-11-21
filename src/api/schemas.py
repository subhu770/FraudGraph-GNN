"""Pydantic schemas and data contracts for FraudGraph-GNN API and streaming engine."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Supported heterogeneous node types in the financial graph."""
    ACCOUNT = "ACCOUNT"
    DEVICE = "DEVICE"
    IP = "IP"


class RelationType(str, Enum):
    """Heterogeneous relation types between financial entities."""
    TRANSFERS_TO = "TRANSFERS_TO"
    TRANSFERS_FROM = "TRANSFERS_FROM"
    USED_DEVICE = "USED_DEVICE"
    DEVICE_OF = "DEVICE_OF"
    ACCESSED_FROM_IP = "ACCESSED_FROM_IP"
    IP_OF = "IP_OF"
    SELF_LOOP = "SELF_LOOP"


class RiskLevel(str, Enum):
    """Categorical risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionEvent(BaseModel):
    """Incoming transaction event stream payload."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    source_account: str = Field(..., description="Originating account ID")
    target_account: str = Field(..., description="Destination account ID")
    amount: float = Field(..., gt=0.0, description="Transaction monetary amount in currency units")
    timestamp: float = Field(..., description="Epoch timestamp in seconds")
    device_id: Optional[str] = Field(None, description="Hardware fingerprint / Device identifier")
    ip_address: Optional[str] = Field(None, description="Client IPv4/IPv6 address")
    currency: str = Field("USD", description="ISO 4217 currency code")
    is_fraud: Optional[bool] = Field(None, description="Ground truth label for simulation / training")
    fraud_pattern: Optional[str] = Field(None, description="Ground truth motif name if synthetic")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "tx_9876543210",
                "source_account": "acc_mule_001",
                "target_account": "acc_mule_002",
                "amount": 9500.0,
                "timestamp": 1716200000.0,
                "device_id": "dev_emulator_xyz",
                "ip_address": "198.51.100.42",
                "currency": "USD",
                "is_fraud": True,
                "fraud_pattern": "CIRCULAR_LAYERING",
            }
        }


class GraphMotifAlert(BaseModel):
    """Explainable graph motif / topological anomaly alert."""
    motif_type: str = Field(..., description="Motif name, e.g. CIRCULAR_LAYERING, SMURFING_FAN_OUT")
    description: str = Field(..., description="Human-readable explanation of why this was flagged")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Rule / detector confidence score")
    involved_nodes: List[str] = Field(default_factory=list, description="IDs of accounts/devices forming the ring")
    severity: RiskLevel = Field(..., description="Severity classification")


class FraudScoreResponse(BaseModel):
    """Real-time transaction fraud score and multi-hop graph explanation."""
    transaction_id: str
    source_account: str
    target_account: str
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud [0.0, 1.0]")
    is_flagged: bool = Field(..., description="True if fraud_score exceeds operational threshold")
    risk_level: RiskLevel = Field(..., description="Risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    inference_latency_ms: float = Field(..., description="Total inference time in milliseconds")
    structural_signals: Dict[str, float] = Field(default_factory=dict, description="Extracted topological features")
    motifs: List[GraphMotifAlert] = Field(default_factory=list, description="Detected money mule and fraud motifs")
    model_version: str = Field("FraudGraph-RGCN-v1.0.0", description="Serving model version")


class SubgraphNode(BaseModel):
    """Serialized node in an extracted ego-subgraph."""
    id: str
    node_type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SubgraphEdge(BaseModel):
    """Serialized directed edge in an extracted ego-subgraph."""
    source: str
    target: str
    relation: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SubgraphResponse(BaseModel):
    """Ego-subgraph representation around an account."""
    account_id: str
    k_hops: int
    num_nodes: int
    num_edges: int
    nodes: List[SubgraphNode]
    edges: List[SubgraphEdge]


class IngestResponse(BaseModel):
    """Response returned upon streaming transaction ingestion."""
    status: str
    transaction_id: str
    graph_updated: bool
    current_nodes: int
    current_edges: int


class HealthResponse(BaseModel):
    """Health and status telemetry for FraudGraph-GNN service."""
    status: str
    version: str
    graph_node_count: int
    graph_edge_count: int
    uptime_seconds: float
    simulation_mode: bool
