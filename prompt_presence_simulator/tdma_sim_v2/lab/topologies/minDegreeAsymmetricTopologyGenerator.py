import random
from typing import List, Type

from tdma_sim_v2.model.nodes.node import Node
from tdma_sim_v2.lab.topologies.topologyGenerator import TopologyGenerator


class MinDegreeAsymmetricTopologyGenerator(TopologyGenerator):

    def __init__(self, degree: int):
        super().__init__(degree)

    def generate(self, node_type: Type[Node], size: int) -> List[Node]:
        """Generates undirected graph topology assuring that every node has a given or higher connection degree"""
        super().generate(node_type, size)
        nodes = [node_type(idx) for idx in range(size)]
        for extended_node in nodes:
            unrelated_nodes = [n for n in nodes if _extension_allowed(extended_node, n)]
            for attached_idx in range(self.degree - len(extended_node.inbound)):
                attached_node = random.choice(unrelated_nodes)
                extended_node.connect(attached_node, bidirect=True)
                unrelated_nodes.remove(attached_node)
        return nodes


def _extension_allowed(receiver: Node, sender: Node) -> bool:
    return sender is not receiver and sender not in receiver.inbound
