"""Structural and temporal graph feature extraction for R-GCN message passing

and non-graph baseline comparisons.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np

from src.api.schemas import NodeType, RelationType, TransactionEvent

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """Computes structural topological signals, localized PageRank,

    degree velocities, cycle counts, and device sharing metrics.
    """

    FEATURE_NAMES: List[str] = [
        "amount_log",
        "src_out_degree",
        "src_in_degree",
        "src_degree_ratio",
        "dst_in_degree",
        "dst_out_degree",
        "src_volume_sent_log",
        "src_volume_recv_log",
        "src_volume_ratio",
        "src_pagerank",
        "dst_pagerank",
        "cycle_count",
        "shared_device_count",
        "shared_ip_count",
        "device_user_count",
        "is_shared_device",
        "out_degree_velocity",
        "in_degree_velocity",
        "time_delta_last_tx",
    ]

    TABULAR_BASELINE_FEATURES: List[str] = [
        "amount",
        "amount_log",
        "hour_of_day",
        "src_tx_count_sent",
        "dst_tx_count_received",
        "src_volume_sent_log",
        "dst_volume_recv_log",
        "is_night_transaction",
    ]

    def extract_structural_signals(
        self,
        event: TransactionEvent,
        ego_graph: nx.DiGraph,
    ) -> Dict[str, float]:
        """Compute structural signals for a transaction within its ego-subgraph."""
        src = event.source_account
        dst = event.target_account
        amount = float(event.amount)
        ts = float(event.timestamp)

        signals: Dict[str, float] = {}

        # 1. Basic monetary signal
        signals["amount_log"] = float(np.log1p(amount))

        # 2. Degree signals
        src_out = float(len([n for n in ego_graph.successors(src) if ego_graph.nodes[n].get("node_type") == NodeType.ACCOUNT.value])) if ego_graph.has_node(src) else 0.0
        src_in = float(len([n for n in ego_graph.predecessors(src) if ego_graph.nodes[n].get("node_type") == NodeType.ACCOUNT.value])) if ego_graph.has_node(src) else 0.0
        dst_in = float(len([n for n in ego_graph.predecessors(dst) if ego_graph.nodes[n].get("node_type") == NodeType.ACCOUNT.value])) if ego_graph.has_node(dst) else 0.0
        dst_out = float(len([n for n in ego_graph.successors(dst) if ego_graph.nodes[n].get("node_type") == NodeType.ACCOUNT.value])) if ego_graph.has_node(dst) else 0.0

        signals["src_out_degree"] = src_out
        signals["src_in_degree"] = src_in
        signals["src_degree_ratio"] = float((src_out + 1.0) / (src_in + 1.0))
        signals["dst_in_degree"] = dst_in
        signals["dst_out_degree"] = dst_out

        # 3. Volume and activity signals
        src_data = ego_graph.nodes.get(src, {}) if ego_graph.has_node(src) else {}
        vol_sent = float(src_data.get("volume_sent", amount))
        vol_recv = float(src_data.get("volume_received", 0.0))
        signals["src_volume_sent_log"] = float(np.log1p(vol_sent))
        signals["src_volume_recv_log"] = float(np.log1p(vol_recv))
        signals["src_volume_ratio"] = float((vol_sent + 1.0) / (vol_recv + 1.0))

        # 4. Localized PageRank
        # Build account-only projection for PageRank
        account_sub = nx.DiGraph()
        for u, v, d in ego_graph.edges(data=True):
            if d.get("relation") == RelationType.TRANSFERS_TO.value:
                account_sub.add_edge(u, v, weight=d.get("tx_count", 1))

        if account_sub.number_of_nodes() > 1:
            try:
                pr = nx.pagerank(account_sub, alpha=0.85, max_iter=30, tol=1e-4)
                signals["src_pagerank"] = float(pr.get(src, 0.0))
                signals["dst_pagerank"] = float(pr.get(dst, 0.0))
            except Exception:
                signals["src_pagerank"] = float(1.0 / max(1, account_sub.number_of_nodes()))
                signals["dst_pagerank"] = float(1.0 / max(1, account_sub.number_of_nodes()))
        else:
            signals["src_pagerank"] = 0.5
            signals["dst_pagerank"] = 0.5

        # 5. Cycle Participation Count
        cycle_count = 0.0
        if account_sub.number_of_nodes() >= 3 and account_sub.has_node(src):
            try:
                # Count small cycles (<= 4 nodes)
                cycles = [c for c in nx.simple_cycles(account_sub, length_bound=4) if src in c]
                cycle_count = float(len(cycles))
            except Exception:
                cycle_count = 0.0
        signals["cycle_count"] = cycle_count

        # 6. Device and IP Collusion Signals
        device_node = f"dev_{event.device_id}" if event.device_id and not event.device_id.startswith("dev_") else event.device_id
        ip_node = f"ip_{event.ip_address}" if event.ip_address and not event.ip_address.startswith("ip_") else event.ip_address

        shared_dev_count = 0.0
        if ego_graph.has_node(src):
            for dev in src_data.get("devices", set()):
                if ego_graph.has_node(dev):
                    dev_data = ego_graph.nodes[dev]
                    linked = dev_data.get("linked_accounts", set())
                    if len(linked) >= 2:
                        shared_dev_count += 1.0
        signals["shared_device_count"] = shared_dev_count

        shared_ip_count = 0.0
        if ego_graph.has_node(src):
            for ip in src_data.get("ips", set()):
                if ego_graph.has_node(ip):
                    ip_data = ego_graph.nodes[ip]
                    linked = ip_data.get("linked_accounts", set())
                    if len(linked) >= 2:
                        shared_ip_count += 1.0
        signals["shared_ip_count"] = shared_ip_count

        dev_user_count = 1.0
        is_shared_dev = 0.0
        if device_node and ego_graph.has_node(device_node):
            dev_data = ego_graph.nodes[device_node]
            linked = dev_data.get("linked_accounts", set())
            dev_user_count = float(len(linked))
            if dev_user_count >= 2:
                is_shared_dev = 1.0
        signals["device_user_count"] = dev_user_count
        signals["is_shared_device"] = is_shared_dev

        # 7. Degree Velocities (transactions per active hour)
        created_at = src_data.get("created_at", ts)
        active_hours = max(0.1, (ts - created_at) / 3600.0)
        signals["out_degree_velocity"] = float(src_out / active_hours)
        signals["in_degree_velocity"] = float(src_in / active_hours)

        last_updated = src_data.get("updated_at", ts)
        signals["time_delta_last_tx"] = float(max(0.0, ts - last_updated))

        return signals

    def extract_tabular_features(
        self,
        event: TransactionEvent,
        ego_graph: Optional[nx.DiGraph] = None,
    ) -> np.ndarray:
        """Extract strictly non-graph, tabular transaction-level features

        for benchmark comparison against the multi-hop GNN model.
        """
        amount = float(event.amount)
        amount_log = float(np.log1p(amount))

        dt = datetime.fromtimestamp(event.timestamp)
        hour = float(dt.hour)
        is_night = 1.0 if (hour < 6 or hour > 22) else 0.0

        src_tx_count = 1.0
        src_vol_log = amount_log
        dst_tx_count = 1.0
        dst_vol_log = amount_log

        if ego_graph is not None:
            if ego_graph.has_node(event.source_account):
                sdata = ego_graph.nodes[event.source_account]
                src_tx_count = float(sdata.get("tx_count_sent", 1))
                src_vol_log = float(np.log1p(sdata.get("volume_sent", amount)))

            if ego_graph.has_node(event.target_account):
                tdata = ego_graph.nodes[event.target_account]
                dst_tx_count = float(tdata.get("tx_count_received", 1))
                dst_vol_log = float(np.log1p(tdata.get("volume_received", amount)))

        features = [
            amount,
            amount_log,
            hour,
            src_tx_count,
            dst_tx_count,
            src_vol_log,
            dst_vol_log,
            is_night,
        ]
        return np.array(features, dtype=np.float32)

    def extract_node_features_matrix(
        self,
        graph: nx.DiGraph,
        node_order: List[str],
    ) -> np.ndarray:
        """Construct a fixed 16-dimensional raw feature vector for every node

        in the extracted ego-graph for R-GCN message passing.
        """
        num_nodes = len(node_order)
        features = np.zeros((num_nodes, 16), dtype=np.float32)

        for idx, node in enumerate(node_order):
            if not graph.has_node(node):
                continue
            data = graph.nodes[node]
            ntype = data.get("node_type", NodeType.ACCOUNT.value)

            # One-hot node type [0..2]
            if ntype == NodeType.ACCOUNT.value:
                features[idx, 0] = 1.0
                in_deg = float(graph.in_degree(node))
                out_deg = float(graph.out_degree(node))
                features[idx, 3] = np.log1p(in_deg)
                features[idx, 4] = np.log1p(out_deg)
                features[idx, 5] = np.log1p(float(data.get("volume_sent", 0.0)))
                features[idx, 6] = np.log1p(float(data.get("volume_received", 0.0)))
                features[idx, 7] = np.log1p(float(data.get("tx_count_sent", 0) + data.get("tx_count_received", 0)))
                features[idx, 10] = float((out_deg + 1.0) / (in_deg + 1.0))
                features[idx, 11] = 1.0 if len(data.get("devices", set())) > 1 else 0.0
                features[idx, 12] = 1.0 if len(data.get("ips", set())) > 1 else 0.0

            elif ntype == NodeType.DEVICE.value:
                features[idx, 1] = 1.0
                linked = float(len(data.get("linked_accounts", set())))
                features[idx, 3] = np.log1p(linked)
                features[idx, 4] = np.log1p(linked)
                features[idx, 7] = np.log1p(float(data.get("tx_count", 1)))
                features[idx, 11] = 1.0 if linked >= 2 else 0.0

            elif ntype == NodeType.IP.value:
                features[idx, 2] = 1.0
                linked = float(len(data.get("linked_accounts", set())))
                features[idx, 3] = np.log1p(linked)
                features[idx, 4] = np.log1p(linked)
                features[idx, 7] = np.log1p(float(data.get("tx_count", 1)))
                features[idx, 12] = 1.0 if linked >= 2 else 0.0

            # Bias term
            features[idx, 15] = 1.0

        return features
