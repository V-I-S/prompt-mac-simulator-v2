from __future__ import annotations

import logging
from typing import List

import tdma_sim_v2.model.messages.message as m
import tdma_sim_v2.model.networks.network as n


class Node:
    logger = logging.getLogger(__name__)

    def __init__(self, node_id: int):
        self.node_id: int = node_id
        self.network = None
        self.inbound: List[Node] = list()
        self.outbound: List[Node] = list()
        self.value: float = 0.0

    def listen(self, node: Node) -> None:
        """Tunes on `self` to listen to `node` messages. I.e. adds `node` to `inbound` messages"""
        if node == self:
            return
        self.inbound.append(node)
        node.outbound.append(self)
        self.logger.debug('Connection %d -> %d created', node.node_id, self.node_id)

    def connect(self, node: Node, bidirect: bool = False) -> None:
        """Creates connection from `self` to `node`, or bidirectional if marked"""
        node.listen(self)
        if bidirect:
            self.listen(node)

    def assign(self, network: 'n.Network'):
        """Update the Node object as soon as the network is known."""
        self.network = network

    def set(self, value: float) -> None:
        """Set the initial or intermediate arbitrary value to Node's voting opinion."""
        self.logger.debug('Node %d has got value: %.3f', self.node_id, value)
        self.value = value

    def trigger(self) -> int:
        """To be triggered by playmaker (executor)."""
        self.logger.debug('Node %d has been triggered', self.node_id)
        raise NotImplementedError

    def vote(self) -> float:
        """Get node voting between [-1.0, 1.0]. Depends upon exact implementation Network is converged when either all nodes return 0.0 or return 1.0."""
        raise NotImplementedError

    def ingest(self, msg: 'm.Message') -> None:
        self.logger.debug('Node %d received message: %s', self.node_id, msg)
        raise NotImplementedError

    def send(self, msg: 'm.MessageBody', addressee: Node) -> None:
        enveloped_msg = m.UnicastMessage(msg, self, addressee)
        self.network.transmit(enveloped_msg)
        self.logger.debug('Node %d sent message: %s', self.node_id, enveloped_msg)

    def announce(self, msg: 'm.MessageBody', addressees: List[Node] = None) -> None:
        addressees = self._adjust_addressees(addressees)
        enveloped_msg = m.MulticastMessage(msg, self, addressees)
        self.network.transmit(enveloped_msg)
        self.logger.debug('Node %d sent message: %s', self.node_id, enveloped_msg)

    def _adjust_addressees(self, original_addressees: List[Node]) -> List[Node]:
        if original_addressees is None:
            return list(self.outbound)
        return original_addressees

    def __str__(self):
        return f'Node {self.node_id}'

    def __eq__(self, other):
        return isinstance(other, Node) and self.node_id == other.node_id
