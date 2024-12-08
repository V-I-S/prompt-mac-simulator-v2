import logging
from functools import reduce
from typing import Optional, List, Tuple

import tdma_sim_v2.model.messages.message as m
import tdma_sim_v2.model.nodes.node as n
from tdma_sim_v2.metrics.networkMetrics import NetworkMetrics


class Network:
    logger = logging.getLogger(__name__)

    def __init__(self, nodes: List['n.Node']):
        self.relay = list()
        self.metrics = NetworkMetrics()
        self.nodes = nodes
        self._update_nodes(nodes)

    def __len__(self):
        return len(self.nodes)

    def reset(self, valuation: List[float]) -> None:
        if len(valuation) != len(self.nodes):
            raise AssertionError('Valuation vector has to got the same size as set of nodes')
        for idx in range(len(valuation)):
            self.nodes[idx].set(valuation[idx])
        self.metrics = NetworkMetrics()

    def get_sentiment(self) -> float:
        return self._count_votes()

    def get_topology_repr(self) -> List[List[int]]:
        topology = [[] for _ in self.nodes]
        for node in self.nodes:
            topology[node.node_id] = [neigh.node_id for neigh in node.outbound]
        return topology

    def get_valuation(self) -> List[float]:
        return [node.value for node in self.nodes]

    def is_communication_success(self, _: int = 0) -> bool:
        votes = self._count_votes()
        self.metrics.record_sentiment(votes)
        return abs(votes) == float(len(self.nodes))

    def is_communication_success_percentile(self, percentile: float) -> bool:
        raise NotImplementedError

    def has_common_tilt(self) -> bool:
        n_tilted_no = self._count_nodes_tilted_toward_no()
        n_tilted_yes = self._count_nodes_tilted_toward_yes()
        return n_tilted_no == 0 or n_tilted_yes == 0

    def transmit(self, msg: 'm.Message') -> None:
        """
        Message transmission - appending to the relay to simulate parallel transmission & execution.
        Nodes shall call this method. It's a good place to count the number of posted communicates in the network.
        """
        self.metrics.count_message()
        self.relay.append(msg)

    def deliver(self) -> Tuple[Optional['m.Message'], bool]:
        """
        Delivery of messages queued in relay ether - may be straight forward or introduce some noise/failures
        Nodes shall never call this method. Delivery of messages is purely network & executor's domain matter.

        @:returns Tuple containing (message: Message, is_converged: bool)
        """
        if not self.relay:
            self.logger.debug('Network relay is empty, sentiment: %.3f', self._count_votes())
            return None, self.is_communication_success()
        msg = self.relay.pop(0)
        if isinstance(msg, m.UnicastMessage):
            self.metrics.count_delivery()
            msg.target.ingest(msg)
        elif isinstance(msg, m.MulticastMessage):
            self.metrics.count_delivery(len(msg.target))
            for node in msg.target:
                node.ingest(msg)
        return msg, self.is_communication_success()

    def _count_votes(self) -> float:
        return reduce((lambda v, w: v + w),
                      map(lambda node: node.vote(), self.nodes))

    def _count_nodes_tilted_toward_no(self) -> int:
        return len(list(filter(lambda node: node.vote() < 0, self.nodes)))

    def _count_nodes_tilted_toward_yes(self) -> int:
        return len(list(filter(lambda node: node.vote() > 0, self.nodes)))

    def _update_nodes(self, nodes: List['n.Node']) -> None:
        if nodes:
            for node in nodes:
                node.assign(self)
