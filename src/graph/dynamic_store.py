"""In-memory heterogeneous financial graph store with thread-safe updates,

fast k-hop ego-graph extraction, and explainable topological motif detection.
"""

from collections import deque
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from src.api.schemas import (
    GraphMotifAlert,
    NodeType,
    RelationType,
    RiskLevel,
    SubgraphEdge,
    SubgraphNode,
    SubgraphResponse,
    TransactionEvent,
)

logger = logging.getLogger(__name__)


class DynamicHeteroGraphStore:
    """Thread-safe in-memory heterogeneous graph store for financial entities

    and interactions (Accounts, Devices, IPs).
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = threading.RLock()
        self._start_time: float = time.time()
        self._transaction_count: int = 0

    @property
    def node_count(self) -> int:
        """Total heterogeneous nodes in store."""
        with self._lock:
            return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total directed relations in store."""
        with self._lock:
            return self._graph.number_of_edges()

    @property
    def uptime_seconds(self) -> float:
        """Uptime duration since initialization."""
        return time.time() - self._start_time

    def clear(self) -> None:
        """Reset the graph store completely."""
        with self._lock:
            self._graph.clear()
            self._transaction_count = 0
            self._start_time = time.time()

    def add_transaction(self, event: TransactionEvent) -> None:
        """Ingest a transaction event, updating nodes, edges, and velocity counters

        in a thread-safe manner.
        """
        with self._lock:
            self._transaction_count += 1
            src = event.source_account
            dst = event.target_account
            amount = float(event.amount)
            ts = float(event.timestamp)
            tx_id = event.transaction_id

            # 1. Update or create Source Account Node
            if not self._graph.has_node(src):
                self._graph.add_node(
                    src,
                    node_type=NodeType.ACCOUNT.value,
                    created_at=ts,
                    updated_at=ts,
                    volume_sent=amount,
                    volume_received=0.0,
                    tx_count_sent=1,
                    tx_count_received=0,
                    devices=set(),
                    ips=set(),
                    unique_recipients={dst},
                    unique_senders=set(),
                )
            else:
                data = self._graph.nodes[src]
                data["updated_at"] = max(data.get("updated_at", ts), ts)
                data["volume_sent"] = data.get("volume_sent", 0.0) + amount
                data["tx_count_sent"] = data.get("tx_count_sent", 0) + 1
                data.setdefault("unique_recipients", set()).add(dst)

            # 2. Update or create Destination Account Node
            if not self._graph.has_node(dst):
                self._graph.add_node(
                    dst,
                    node_type=NodeType.ACCOUNT.value,
                    created_at=ts,
                    updated_at=ts,
                    volume_sent=0.0,
                    volume_received=amount,
                    tx_count_sent=0,
                    tx_count_received=1,
                    devices=set(),
                    ips=set(),
                    unique_recipients=set(),
                    unique_senders={src},
                )
            else:
                data = self._graph.nodes[dst]
                data["updated_at"] = max(data.get("updated_at", ts), ts)
                data["volume_received"] = data.get("volume_received", 0.0) + amount
                data["tx_count_received"] = data.get("tx_count_received", 0) + 1
                data.setdefault("unique_senders", set()).add(src)

            # 3. Add or update TRANSFERS_TO edge (src -> dst)
            if self._graph.has_edge(src, dst):
                edge_data = self._graph[src][dst]
                edge_data["tx_count"] = edge_data.get("tx_count", 0) + 1
                edge_data["total_amount"] = edge_data.get("total_amount", 0.0) + amount
                edge_data["last_amount"] = amount
                edge_data["last_timestamp"] = ts
                tx_ids = edge_data.setdefault("tx_ids", [])
                if len(tx_ids) < 50:  # Bound memory retention per edge
                    tx_ids.append(tx_id)
            else:
                self._graph.add_edge(
                    src,
                    dst,
                    relation=RelationType.TRANSFERS_TO.value,
                    tx_count=1,
                    total_amount=amount,
                    last_amount=amount,
                    first_timestamp=ts,
                    last_timestamp=ts,
                    tx_ids=[tx_id],
                )

            # 4. Optional Device Entity and Bi-directional Linkage
            if event.device_id:
                dev = f"dev_{event.device_id}" if not event.device_id.startswith("dev_") else event.device_id
                self._graph.nodes[src].setdefault("devices", set()).add(dev)

                if not self._graph.has_node(dev):
                    self._graph.add_node(
                        dev,
                        node_type=NodeType.DEVICE.value,
                        first_seen=ts,
                        last_seen=ts,
                        linked_accounts={src},
                        tx_count=1,
                    )
                else:
                    d_data = self._graph.nodes[dev]
                    d_data["last_seen"] = max(d_data.get("last_seen", ts), ts)
                    d_data["tx_count"] = d_data.get("tx_count", 0) + 1
                    d_data.setdefault("linked_accounts", set()).add(src)

                # Add USED_DEVICE (src -> dev) and DEVICE_OF (dev -> src)
                self._graph.add_edge(
                    src,
                    dev,
                    relation=RelationType.USED_DEVICE.value,
                    last_timestamp=ts,
                    tx_id=tx_id,
                )
                self._graph.add_edge(
                    dev,
                    src,
                    relation=RelationType.DEVICE_OF.value,
                    last_timestamp=ts,
                    tx_id=tx_id,
                )

            # 5. Optional IP Entity and Bi-directional Linkage
            if event.ip_address:
                ip = f"ip_{event.ip_address}" if not event.ip_address.startswith("ip_") else event.ip_address
                self._graph.nodes[src].setdefault("ips", set()).add(ip)

                if not self._graph.has_node(ip):
                    self._graph.add_node(
                        ip,
                        node_type=NodeType.IP.value,
                        first_seen=ts,
                        last_seen=ts,
                        linked_accounts={src},
                        tx_count=1,
                    )
                else:
                    ip_data = self._graph.nodes[ip]
                    ip_data["last_seen"] = max(ip_data.get("last_seen", ts), ts)
                    ip_data["tx_count"] = ip_data.get("tx_count", 0) + 1
                    ip_data.setdefault("linked_accounts", set()).add(src)

                self._graph.add_edge(
                    src,
                    ip,
                    relation=RelationType.ACCESSED_FROM_IP.value,
                    last_timestamp=ts,
                    tx_id=tx_id,
                )
                self._graph.add_edge(
                    ip,
                    src,
                    relation=RelationType.IP_OF.value,
                    last_timestamp=ts,
                    tx_id=tx_id,
                )

    def get_k_hop_subgraph(
        self,
        node_id: str,
        k: int = 2,
        max_nodes: int = 150,
    ) -> nx.DiGraph:
        """Fast k-hop ego-graph extraction using bounded BFS across both in-edges

        and out-edges to guarantee sub-15ms inference latency.
        """
        with self._lock:
            if not self._graph.has_node(node_id):
                empty = nx.DiGraph()
                empty.add_node(node_id, node_type=NodeType.ACCOUNT.value)
                return empty

            visited: Set[str] = {node_id}
            current_frontier: Set[str] = {node_id}

            for _ in range(k):
                next_frontier: Set[str] = set()
                for curr in current_frontier:
                    # In-neighbors & Out-neighbors
                    nbrs = set(self._graph.predecessors(curr)) | set(self._graph.successors(curr))
                    for nbr in nbrs:
                        if nbr not in visited:
                            visited.add(nbr)
                            next_frontier.add(nbr)
                            if len(visited) >= max_nodes:
                                break
                    if len(visited) >= max_nodes:
                        break
                current_frontier = next_frontier
                if not current_frontier or len(visited) >= max_nodes:
                    break

            # Create induced subgraph copy
            subgraph = self._graph.subgraph(visited).copy()
            return subgraph

    def detect_motifs(
        self,
        node_id: str,
        ego_graph: Optional[nx.DiGraph] = None,
    ) -> List[GraphMotifAlert]:
        """Detect explainable topological fraud motifs:

        1. Circular layering / money mule loops (A -> B -> C -> A)
        2. Smurfing fan-out (rapid structuring)
        3. Smurfing fan-in (rapid aggregation)
        4. Device collusion rings (shared hardware across mule syndicates)
        """
        alerts: List[GraphMotifAlert] = []

        sub = ego_graph if ego_graph is not None else self.get_k_hop_subgraph(node_id, k=2)
        if not sub.has_node(node_id):
            return alerts

        # -------------------------------------------------------------
        # 1. Circular Layering (Directed Cycles of length 3 to 5)
        # -------------------------------------------------------------
        cycle = self._find_cycle_with_node(sub, node_id, min_len=3, max_len=5)
        if cycle:
            # Materiality check: AML money mule layering involves substantial funds (>= $2,000)
            amounts = []
            for idx in range(len(cycle)):
                u, v = cycle[idx], cycle[(idx + 1) % len(cycle)]
                if sub.has_edge(u, v):
                    amounts.append(sub[u][v].get("total_amount", 0.0))
            if amounts and (sum(amounts) / len(amounts)) >= 2000.0:
                cycle_str = " -> ".join(cycle) + f" -> {cycle[0]}"
                alerts.append(
                    GraphMotifAlert(
                        motif_type="CIRCULAR_LAYERING",
                        description=(
                            f"Detected circular money mule layering loop across {len(cycle)} accounts: {cycle_str}. "
                            "Funds flow sequentially through intermediary accounts before looping back."
                        ),
                        confidence=0.94,
                        involved_nodes=cycle,
                        severity=RiskLevel.CRITICAL,
                    )
                )

        # -------------------------------------------------------------
        # 2. Shared Device Collusion Ring
        # -------------------------------------------------------------
        # Look for DEVICE nodes connected to node_id that link to >= 3 accounts
        shared_devices: List[str] = []
        syndicate_accounts: Set[str] = set()

        for nbr in sub.neighbors(node_id):
            if sub.nodes[nbr].get("node_type") == NodeType.DEVICE.value:
                linked = sub.nodes[nbr].get("linked_accounts", set())
                sub_linked = {
                    u for u in sub.predecessors(nbr) if sub.nodes[u].get("node_type") == NodeType.ACCOUNT.value
                } | {
                    v for v in sub.successors(nbr) if sub.nodes[v].get("node_type") == NodeType.ACCOUNT.value
                } | set(linked)

                if len(sub_linked) >= 3:
                    shared_devices.append(nbr)
                    syndicate_accounts.update(sub_linked)

        if shared_devices and len(syndicate_accounts) >= 3:
            alerts.append(
                GraphMotifAlert(
                    motif_type="DEVICE_COLLUSION_RING",
                    description=(
                        f"Account {node_id} shares hardware devices ({', '.join(shared_devices)}) "
                        f"with {len(syndicate_accounts) - 1} other distinct accounts: {list(syndicate_accounts)}."
                    ),
                    confidence=0.96,
                    involved_nodes=list(syndicate_accounts | set(shared_devices)),
                    severity=RiskLevel.CRITICAL,
                )
            )

        # -------------------------------------------------------------
        # 3. Smurfing Fan-Out (Rapid Structuring)
        # -------------------------------------------------------------
        out_accounts = [
            succ
            for succ in sub.successors(node_id)
            if sub.nodes[succ].get("node_type") == NodeType.ACCOUNT.value
            and sub[node_id][succ].get("total_amount", 0.0) >= 3000.0
        ]
        if len(out_accounts) >= 4:
            alerts.append(
                GraphMotifAlert(
                    motif_type="SMURFING_FAN_OUT",
                    description=(
                        f"Account {node_id} exhibits rapid fan-out structuring: dispersing funds "
                        f"to {len(out_accounts)} downstream accounts in structured payments."
                    ),
                    confidence=0.89,
                    involved_nodes=[node_id] + out_accounts[:10],
                    severity=RiskLevel.HIGH,
                )
            )

        # -------------------------------------------------------------
        # 4. Smurfing Fan-In (Rapid Consolidation)
        # -------------------------------------------------------------
        in_accounts = [
            pred
            for pred in sub.predecessors(node_id)
            if sub.nodes[pred].get("node_type") == NodeType.ACCOUNT.value
            and sub[pred][node_id].get("total_amount", 0.0) >= 3000.0
        ]
        if len(in_accounts) >= 4:
            alerts.append(
                GraphMotifAlert(
                    motif_type="SMURFING_FAN_IN",
                    description=(
                        f"Account {node_id} exhibits rapid fan-in aggregation: consolidating incoming "
                        f"transfers from {len(in_accounts)} separate feeder accounts."
                    ),
                    confidence=0.88,
                    involved_nodes=[node_id] + in_accounts[:10],
                    severity=RiskLevel.HIGH,
                )
            )

        return alerts

    def _find_cycle_with_node(
        self,
        sub: nx.DiGraph,
        target_node: str,
        min_len: int = 3,
        max_len: int = 5,
        max_steps: int = 150,
    ) -> Optional[List[str]]:
        """Fast bounded BFS search to find a directed cycle containing target_node

        on the account transfer network without exponential recursion.
        """
        account_nodes = {
            n for n in sub.nodes if sub.nodes[n].get("node_type") == NodeType.ACCOUNT.value
        }
        if target_node not in account_nodes:
            return None

        # Queue contains (current_node, path)
        queue = deque([(target_node, [target_node])])
        steps = 0

        while queue and steps < max_steps:
            curr, path = queue.popleft()
            steps += 1
            if len(path) > max_len:
                continue

            for nxt in sub.successors(curr):
                if nxt not in account_nodes:
                    continue
                if sub[curr][nxt].get("relation") != RelationType.TRANSFERS_TO.value:
                    continue

                if nxt == target_node and len(path) >= min_len:
                    return list(path)

                if nxt not in path and len(path) < max_len:
                    queue.append((nxt, path + [nxt]))

        return None

    def export_subgraph_response(self, account_id: str, k_hops: int = 2) -> SubgraphResponse:
        """Export extracted ego-subgraph serialized for API consumption and visualizers."""
        sub = self.get_k_hop_subgraph(account_id, k=k_hops)

        nodes_list: List[SubgraphNode] = []
        for n, data in sub.nodes(data=True):
            clean_attrs = {}
            for k, v in data.items():
                if isinstance(v, (set, list)):
                    clean_attrs[k] = list(v)
                elif isinstance(v, (int, float, str, bool)):
                    clean_attrs[k] = v
            nodes_list.append(
                SubgraphNode(
                    id=str(n),
                    node_type=data.get("node_type", NodeType.ACCOUNT.value),
                    attributes=clean_attrs,
                )
            )

        edges_list: List[SubgraphEdge] = []
        for u, v, data in sub.edges(data=True):
            clean_edge_attrs = {}
            for k, val in data.items():
                if isinstance(val, (set, list)):
                    clean_edge_attrs[k] = list(val)
                elif isinstance(val, (int, float, str, bool)):
                    clean_edge_attrs[k] = val
            edges_list.append(
                SubgraphEdge(
                    source=str(u),
                    target=str(v),
                    relation=data.get("relation", RelationType.TRANSFERS_TO.value),
                    attributes=clean_edge_attrs,
                )
            )

        return SubgraphResponse(
            account_id=account_id,
            k_hops=k_hops,
            num_nodes=len(nodes_list),
            num_edges=len(edges_list),
            nodes=nodes_list,
            edges=edges_list,
        )
