from random import randrange

from tdma_sim_v2.model.messages.vectorMessageBody import VectorMessageBody
from tdma_sim_v2.model.nodes.node import Node


class tSlotsNode(Node):

    def vote(self) -> float:
        pass

    def __init__(self, node_id: int, n_slots: int, n_selects: int):
        super().__init__(node_id)
        self.occupied_slots = []
        self.available_slots = n_slots
        self.transmission_probability = n_selects

    def trigger(self) -> int:
        untouched_slots = list(range(self.available_slots))
        for slt in range(self.transmission_probability):
            self.occupied_slots += [untouched_slots.pop(randrange(0, len(untouched_slots)))]
        self.announce(VectorMessageBody(self.occupied_slots))
        return len(self.occupied_slots)
