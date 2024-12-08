import random
from typing import List, Type

from tdma_sim_v2.model.nodes.node import Node
from tdma_sim_v2.lab.topologies.topologyGenerator import TopologyGenerator


class ContinuousConstInboundSizeTopologyGenerator(TopologyGenerator):

    def __init__(self, degree: int):
        super().__init__(degree)

    def generate(self, node_type: Type[Node], size: int) -> List[Node]:
        """
        Generates graph topology assuring that every node listens to strictly defined number of neighbors. Number of node's listeners is arbitrary.

        Differs from ``ConstInboundSizeTopologyGenerator``, having a property that each node has at least one listener. So the message stream can be continuous, as each node will
        get some find receiver for its communicate.
        """
        super().generate(node_type, size)
        nodes = [node_type(idx) for idx in range(size)]
        self.cast_output_connections(nodes)
        self.cast_input_connections(nodes)
        return nodes

    def cast_output_connections(self, nodes: List[Node]) -> None:
        for extended_node in nodes:
            expansion_set = [n for n in nodes if self._output_extension_allowed(extended_node, n)]
            if not expansion_set:
                raise AssertionError('Cannot create continuous connection between nodes.')
            listening_node = random.choice(expansion_set)
            listening_node.listen(extended_node)

    def cast_input_connections(self, nodes: List[Node]) -> None:
        for extended_node in nodes:
            expansion_set = [n for n in nodes if self._input_extension_allowed(extended_node, n)]
            for attached_idx in range(self.degree - len(extended_node.inbound)):
                listened_node = random.choice(expansion_set)
                extended_node.listen(listened_node)
                expansion_set.remove(listened_node)

    def _output_extension_allowed(self, sender: Node, receiver: Node) -> bool:
        return sender is not receiver \
               and len(receiver.inbound) < self.degree \
               and sender not in receiver.inbound

    def _input_extension_allowed(self, receiver: Node, sender: Node) -> bool:
        return sender is not receiver \
               and len(receiver.inbound) < self.degree \
               and sender not in receiver.inbound
