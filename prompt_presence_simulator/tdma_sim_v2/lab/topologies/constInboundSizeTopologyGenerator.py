import random
from typing import List, Type

from tdma_sim_v2.model.nodes.node import Node
from tdma_sim_v2.lab.topologies.topologyGenerator import TopologyGenerator


class ConstInboundSizeTopologyGenerator(TopologyGenerator):

    def __init__(self, degree: int):
        super().__init__(degree)

    def generate(self, node_type: Type[Node], size: int) -> List[Node]:
        """Generates graph topology assuring that every node listens to strictly defined number of neighbors. Number of node's listeners is arbitrary."""
        super().generate(node_type, size)
        nodes = [node_type(idx) for idx in range(size)]
        for extended_node in nodes:
            unrelated_nodes = [n for n in nodes if _extension_allowed(extended_node, n)]
            for attached_idx in range(self.degree):
                listened_node = random.choice(unrelated_nodes)
                extended_node.listen(listened_node)
                unrelated_nodes.remove(listened_node)
        return nodes


def _extension_allowed(receiver: Node, sender: Node) -> bool:
    return sender is not receiver and sender not in receiver.inbound
