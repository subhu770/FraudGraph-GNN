"""Deterministic financial transaction stream and fraud ring simulator.

Synthesizes normal retail banking flows alongside injected money mule rings:
1. Circular Layering (A -> B -> C -> A multi-hop cycles)
2. Smurfing Fan-Out / Fan-In (Structuring & rapid aggregation)
3. Device Collusion Rings (Syndicate burner accounts sharing hardware/IPs)
"""

import logging
import random
from typing import Generator, List, Tuple
import numpy as np

from src.api.schemas import TransactionEvent

logger = logging.getLogger(__name__)


class FinancialNetworkSimulator:
    """Generates deterministic, labeled financial networks for real-time ingestion

    benchmarking and GNN vs Tabular comparative evaluation.
    """

    def __init__(
        self,
        num_transactions: int = 5000,
        num_rings: int = 40,
        seed: int = 42,
        base_timestamp: float = 1716200000.0,
    ) -> None:
        self.num_transactions = num_transactions
        self.num_rings = num_rings
        self.seed = seed
        self.base_timestamp = base_timestamp
        self._rng = np.random.RandomState(seed)
        random.seed(seed)

    def generate_dataset(self) -> List[TransactionEvent]:
        """Generate full synthetic dataset with chronological ordering."""
        transactions: List[TransactionEvent] = []
        tx_counter = 0

        # Allocation of rings across the 3 core topologies
        num_circular = int(self.num_rings * 0.375)  # 15 rings
        num_smurfing = int(self.num_rings * 0.375)  # 15 rings
        num_device_rings = self.num_rings - num_circular - num_smurfing  # 10 rings

        # Time range spanning 5 days (432,000 seconds)
        duration_window = 432000.0

        # -------------------------------------------------------------
        # 1. Circular Layering Rings (Cycles of 3 to 5 accounts)
        # -------------------------------------------------------------
        for ring_idx in range(num_circular):
            ring_len = self._rng.randint(3, 6)
            ring_accounts = [f"acc_mule_circ_{ring_idx:02d}_{i}" for i in range(ring_len)]
            initial_amount = float(self._rng.uniform(8500.0, 18000.0))
            ring_start_time = self.base_timestamp + float(self._rng.uniform(100.0, duration_window - 15000.0))

            # Money hops through the cycle with a 3-6% laundering haircut per hop
            curr_amt = initial_amount
            hop_time = ring_start_time

            for hop in range(ring_len):
                src = ring_accounts[hop]
                dst = ring_accounts[(hop + 1) % ring_len]
                tx_id = f"tx_circ_{ring_idx:02d}_{hop}_{tx_counter}"
                tx_counter += 1

                # Rapid succession: 10 to 30 mins between hops
                hop_time += float(self._rng.uniform(600.0, 1800.0))
                # Laundering haircut
                curr_amt *= float(self._rng.uniform(0.94, 0.97))

                dev_id = f"dev_circ_{ring_idx:02d}_{hop}"
                ip_addr = f"198.51.{100 + ring_idx}.{10 + hop}"

                transactions.append(
                    TransactionEvent(
                        transaction_id=tx_id,
                        source_account=src,
                        target_account=dst,
                        amount=round(curr_amt, 2),
                        timestamp=hop_time,
                        device_id=dev_id,
                        ip_address=ip_addr,
                        currency="USD",
                        is_fraud=True,
                        fraud_pattern="CIRCULAR_LAYERING",
                    )
                )

        # -------------------------------------------------------------
        # 2. Smurfing Fan-Out & Fan-In Rings (Structuring)
        # -------------------------------------------------------------
        fan_out_count = num_smurfing // 2
        fan_in_count = num_smurfing - fan_out_count

        # 2a. Smurfing Fan-Out (Kingpin -> many mules)
        for fo_idx in range(fan_out_count):
            kingpin = f"acc_kingpin_fo_{fo_idx:02d}"
            num_smurfs = self._rng.randint(5, 9)
            fo_start_time = self.base_timestamp + float(self._rng.uniform(100.0, duration_window - 15000.0))

            for s_idx in range(num_smurfs):
                mule = f"acc_smurf_mule_{fo_idx:02d}_{s_idx}"
                # Structured just below $10,000 Bank Secrecy Act / AML reporting threshold
                amt = float(self._rng.uniform(8900.0, 9950.0))
                tx_time = fo_start_time + float(self._rng.uniform(120.0, 3600.0))
                tx_id = f"tx_smurf_fo_{fo_idx:02d}_{s_idx}_{tx_counter}"
                tx_counter += 1

                transactions.append(
                    TransactionEvent(
                        transaction_id=tx_id,
                        source_account=kingpin,
                        target_account=mule,
                        amount=round(amt, 2),
                        timestamp=tx_time,
                        device_id=f"dev_fo_{fo_idx:02d}",
                        ip_address=f"203.0.113.{20 + fo_idx}",
                        currency="USD",
                        is_fraud=True,
                        fraud_pattern="SMURFING_FAN_OUT",
                    )
                )

        # 2b. Smurfing Fan-In (Many mules -> Single consolidation account)
        for fi_idx in range(fan_in_count):
            collector = f"acc_collector_fi_{fi_idx:02d}"
            num_feeders = self._rng.randint(5, 9)
            fi_start_time = self.base_timestamp + float(self._rng.uniform(100.0, duration_window - 15000.0))

            for feeder_idx in range(num_feeders):
                feeder = f"acc_feeder_fi_{fi_idx:02d}_{feeder_idx}"
                amt = float(self._rng.uniform(8800.0, 9900.0))
                tx_time = fi_start_time + float(self._rng.uniform(180.0, 4200.0))
                tx_id = f"tx_smurf_fi_{fi_idx:02d}_{feeder_idx}_{tx_counter}"
                tx_counter += 1

                transactions.append(
                    TransactionEvent(
                        transaction_id=tx_id,
                        source_account=feeder,
                        target_account=collector,
                        amount=round(amt, 2),
                        timestamp=tx_time,
                        device_id=f"dev_fi_{fi_idx:02d}_{feeder_idx}",
                        ip_address=f"192.0.2.{40 + feeder_idx}",
                        currency="USD",
                        is_fraud=True,
                        fraud_pattern="SMURFING_FAN_IN",
                    )
                )

        # -------------------------------------------------------------
        # 3. Device Collusion Rings (Shared Emulator / Hardware)
        # -------------------------------------------------------------
        for dev_idx in range(num_device_rings):
            shared_device = f"dev_syndicate_hw_{dev_idx:02d}"
            shared_ip = f"198.18.{dev_idx}.100"
            ring_size = self._rng.randint(4, 7)
            accounts = [f"acc_dev_mule_{dev_idx:02d}_{k}" for k in range(ring_size)]
            drop_account = f"acc_cashout_drop_{dev_idx:02d}"
            dev_start_time = self.base_timestamp + float(self._rng.uniform(100.0, duration_window - 15000.0))

            # Internal transfers and cash-out transfers using same hardware
            for k in range(ring_size):
                src = accounts[k]
                dst = accounts[(k + 1) % ring_size] if k < ring_size - 1 else drop_account
                amt = float(self._rng.uniform(3500.0, 9800.0))
                tx_time = dev_start_time + float(self._rng.uniform(300.0, 5400.0))
                tx_id = f"tx_dev_ring_{dev_idx:02d}_{k}_{tx_counter}"
                tx_counter += 1

                transactions.append(
                    TransactionEvent(
                        transaction_id=tx_id,
                        source_account=src,
                        target_account=dst,
                        amount=round(amt, 2),
                        timestamp=tx_time,
                        device_id=shared_device,
                        ip_address=shared_ip,
                        currency="USD",
                        is_fraud=True,
                        fraud_pattern="SHARED_DEVICE_RING",
                    )
                )

        # -------------------------------------------------------------
        # 4. Legitimate Retail Banking Transactions
        # -------------------------------------------------------------
        num_fraud_txs = len(transactions)
        num_legit_needed = max(500, self.num_transactions - num_fraud_txs)

        # Retail account pool
        num_retail_accounts = 600
        retail_accounts = [f"acc_retail_{i:04d}" for i in range(num_retail_accounts)]
        retail_devices = [f"dev_retail_{i:04d}" for i in range(num_retail_accounts)]

        # Time range spanning 5 days (432,000 seconds)
        duration_window = 432000.0

        for legit_idx in range(num_legit_needed):
            src_idx = self._rng.randint(0, num_retail_accounts)
            dst_idx = self._rng.randint(0, num_retail_accounts)
            while dst_idx == src_idx:
                dst_idx = self._rng.randint(0, num_retail_accounts)

            src = retail_accounts[src_idx]
            dst = retail_accounts[dst_idx]

            # Log-normal distribution around $85 with standard deviation
            amt = float(np.exp(self._rng.normal(4.44, 0.9)))
            amt = max(5.0, min(amt, 1500.0))  # Retail bounds

            tx_time = self.base_timestamp + float(self._rng.uniform(0.0, duration_window))
            tx_id = f"tx_legit_{legit_idx:05d}_{tx_counter}"
            tx_counter += 1

            transactions.append(
                TransactionEvent(
                    transaction_id=tx_id,
                    source_account=src,
                    target_account=dst,
                    amount=round(amt, 2),
                    timestamp=tx_time,
                    device_id=retail_devices[src_idx],
                    ip_address=f"10.{src_idx // 256}.{src_idx % 256}.{(legit_idx % 250) + 1}",
                    currency="USD",
                    is_fraud=False,
                    fraud_pattern="NORMAL",
                )
            )

        # Sort strictly by timestamp for realistic streaming timeline
        transactions.sort(key=lambda t: t.timestamp)
        logger.info(
            f"Generated {len(transactions)} synthetic transactions "
            f"({num_fraud_txs} fraudulent across {self.num_rings} rings, {num_legit_needed} legitimate)"
        )
        return transactions

    def stream_transactions(self) -> Generator[TransactionEvent, None, None]:
        """Yield transactions one-by-one as an ingestion stream."""
        for tx in self.generate_dataset():
            yield tx
