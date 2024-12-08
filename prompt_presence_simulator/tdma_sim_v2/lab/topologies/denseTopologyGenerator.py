from typing import List, Type

from tdma_sim_v2.model.nodes.node import Node
from tdma_sim_v2.lab.topologies.topologyGenerator import TopologyGenerator


class DenseTopologyGenerator(TopologyGenerator):

    def __init__(self, degree: int):
        super().__init__(degree)

    def generate(self, node_type: Type[Node], size: int) -> List[Node]:
        """Generates dense all-to-all connection graph topology"""
        super().generate(node_type, size)
        nodes = [node_type(idx) for idx in range(size)]
        for node in nodes:
            for neighbor in nodes:
                if neighbor != node:
                    node.listen(neighbor)
        return nodes
