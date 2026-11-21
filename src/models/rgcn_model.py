"""Relational Graph Convolutional Network (R-GCN) & GraphSAGE message passing

implemented in pure PyTorch + NetworkX for high-throughput, low-latency CPU scoring.
"""

import os
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.api.schemas import NodeType, RelationType

# Canonical relation indices for tensor routing
RELATION_MAP: Dict[str, int] = {
    RelationType.TRANSFERS_TO.value: 0,
    RelationType.TRANSFERS_FROM.value: 1,
    RelationType.USED_DEVICE.value: 2,
    RelationType.DEVICE_OF.value: 3,
    RelationType.ACCESSED_FROM_IP.value: 4,
    RelationType.IP_OF.value: 5,
    RelationType.SELF_LOOP.value: 6,
}
NUM_RELATIONS: int = len(RELATION_MAP)


class RelationalGraphConvLayer(nn.Module):
    """Single Relational GCN layer with relation-specific projection matrices,

    degree normalization, layer norm, residual link, and dropout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int = NUM_RELATIONS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        # Relation-specific transformation matrices W_r
        self.relation_weights = nn.Parameter(
            torch.empty(num_relations, in_features, out_features)
        )
        # Self-loop transformation matrix W_0
        self.self_weight = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # Residual shortcut if dimensions match
        self.residual = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features, bias=False)
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Glorot / Xavier uniform weight initialization."""
        for r in range(self.num_relations):
            nn.init.xavier_uniform_(self.relation_weights[r])
        nn.init.xavier_uniform_(self.self_weight.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Vectorized message-passing over heterogeneous edge relations.

        Args:
            x: Node feature tensor (N, in_features)
            edge_index_dict: Mapping relation_idx -> Tensor of shape (2, E_r)
        Returns:
            Updated node representations (N, out_features)
        """
        N = x.size(0)
        device = x.device

        # Aggregator initialized to zero
        aggregated = torch.zeros(N, self.out_features, device=device)

        # Message passing per relation type
        for r in range(self.num_relations):
            if r not in edge_index_dict:
                continue
            edges = edge_index_dict[r]
            if edges.numel() == 0 or edges.size(1) == 0:
                continue

            src_idx = edges[0]
            dst_idx = edges[1]
            E_r = edges.size(1)

            # Transform source node features using relation weight W_r: (E_r, out_features)
            src_features = x[src_idx]
            W_r = self.relation_weights[r]  # (in_features, out_features)
            messages = torch.matmul(src_features, W_r)

            # Degree normalization factor (in-degree per relation)
            deg = torch.zeros(N, device=device)
            deg.index_add_(0, dst_idx, torch.ones(E_r, device=device))
            deg = deg.clamp(min=1.0).unsqueeze(1)

            # Scatter-add messages into destination nodes
            rel_agg = torch.zeros(N, self.out_features, device=device)
            rel_agg.index_add_(0, dst_idx, messages)
            rel_agg = rel_agg / deg

            aggregated = aggregated + rel_agg

        # Combine with self-loop
        self_messages = self.self_weight(x)
        out = aggregated + self_messages + self.bias

        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Residual shortcut
        out = out + self.residual(x)
        return out


class RGCNFraudDetector(nn.Module):
    """Multi-layer Relational Graph Convolutional Network with edge-interaction

    classification head for real-time transaction scoring.
    """

    def __init__(
        self,
        node_in_dim: int = 16,
        hidden_dim: int = 64,
        edge_feature_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_layers = num_layers

        # Node feature projection encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Relational Message-Passing Layers
        self.layers = nn.ModuleList([
            RelationalGraphConvLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                num_relations=NUM_RELATIONS,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Link/Edge Fraud Classifier Head
        # Concatenates: [h_src, h_dst, |h_src - h_dst|, h_src * h_dst, edge_features]
        clf_in_dim = (4 * hidden_dim) + edge_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward_embeddings(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Run GNN encoder and relational message passing to compute node embeddings."""
        h = self.node_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return h

    def score_edge(
        self,
        h: torch.Tensor,
        src_idx: int,
        dst_idx: int,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify transaction link probability given node representations."""
        h_src = h[src_idx : src_idx + 1]
        h_dst = h[dst_idx : dst_idx + 1]

        diff = torch.abs(h_src - h_dst)
        prod = h_src * h_dst

        parts = [h_src, h_dst, diff, prod]
        if edge_features is not None:
            if edge_features.dim() == 1:
                edge_features = edge_features.unsqueeze(0)
            parts.append(edge_features)
        else:
            dummy_edge = torch.zeros(1, self.edge_feature_dim, device=h.device)
            parts.append(dummy_edge)

        combined = torch.cat(parts, dim=-1)
        logit = self.classifier(combined)
        prob = torch.sigmoid(logit)
        return prob.squeeze()

    def score_subgraph(
        self,
        graph: nx.DiGraph,
        source_id: str,
        target_id: str,
        node_features_matrix: np.ndarray,
        node_order: List[str],
        edge_features: Optional[np.ndarray] = None,
    ) -> float:
        """Complete CPU-optimized forward inference directly from NetworkX ego-graph."""
        self.eval()
        with torch.no_grad():
            node_map = {n: i for i, n in enumerate(node_order)}
            if source_id not in node_map or target_id not in node_map:
                return 0.05

            src_idx = node_map[source_id]
            dst_idx = node_map[target_id]

            x_tensor = torch.from_numpy(node_features_matrix).float()

            # Build relation edge indices
            edge_dict: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(NUM_RELATIONS)}

            for u, v, data in graph.edges(data=True):
                if u not in node_map or v not in node_map:
                    continue
                u_i = node_map[u]
                v_i = node_map[v]
                rel = data.get("relation", RelationType.TRANSFERS_TO.value)

                if rel == RelationType.TRANSFERS_TO.value:
                    edge_dict[RELATION_MAP[RelationType.TRANSFERS_TO.value]].append((u_i, v_i))
                    # Add reverse edge
                    edge_dict[RELATION_MAP[RelationType.TRANSFERS_FROM.value]].append((v_i, u_i))
                elif rel == RelationType.USED_DEVICE.value:
                    edge_dict[RELATION_MAP[RelationType.USED_DEVICE.value]].append((u_i, v_i))
                    edge_dict[RELATION_MAP[RelationType.DEVICE_OF.value]].append((v_i, u_i))
                elif rel == RelationType.ACCESSED_FROM_IP.value:
                    edge_dict[RELATION_MAP[RelationType.ACCESSED_FROM_IP.value]].append((u_i, v_i))
                    edge_dict[RELATION_MAP[RelationType.IP_OF.value]].append((v_i, u_i))

            # Convert to PyTorch tensors
            tensor_edge_dict: Dict[int, torch.Tensor] = {}
            for r, pairs in edge_dict.items():
                if pairs:
                    tensor_edge_dict[r] = torch.tensor(pairs, dtype=torch.long).t().contiguous()
                else:
                    tensor_edge_dict[r] = torch.empty((2, 0), dtype=torch.long)

            edge_feat_tensor = (
                torch.from_numpy(edge_features).float()
                if edge_features is not None
                else None
            )

            # Forward pass
            h = self.forward_embeddings(x_tensor, tensor_edge_dict)
            prob = self.score_edge(h, src_idx, dst_idx, edge_feat_tensor)
            return float(prob.item())

    def save_pretrained(self, path: str) -> None:
        """Persist model architecture and weights to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "node_in_dim": self.node_in_dim,
                "hidden_dim": self.hidden_dim,
                "edge_feature_dim": self.edge_feature_dim,
                "num_layers": self.num_layers,
            },
            path,
        )

    @classmethod
    def load_pretrained(cls, path: str) -> "RGCNFraudDetector":
        """Load pretrained checkpoint from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            node_in_dim=checkpoint["node_in_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            edge_feature_dim=checkpoint["edge_feature_dim"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
