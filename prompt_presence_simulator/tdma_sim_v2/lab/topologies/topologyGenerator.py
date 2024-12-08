from typing import List, Type

from tdma_sim_v2.model.nodes.node import Node


class TopologyGenerator:
    def __init__(self, degree: int):
        """:degree: refers to the topology evolution step"""
        self.degree = degree

    def generate(self, node_type: Type[Node], size: int) -> List[Node]:
        if size < 0:
            raise AssertionError("Size of the network has to be non-negative")
