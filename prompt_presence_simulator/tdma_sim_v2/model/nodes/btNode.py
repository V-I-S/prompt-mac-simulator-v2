from random import randrange

from tdma_sim_v2.config.execution import BERNOULLI_TRIES_RESOLUTION
from tdma_sim_v2.model.messages.vectorMessageBody import VectorMessageBody
from tdma_sim_v2.model.nodes.tSlotsNode import tSlotsNode


class BtNode(tSlotsNode):

    def vote(self) -> float:
        pass

    def __init__(self, node_id: int, n_slots: int, transmission_probability: int):
        super().__init__(node_id, n_slots, transmission_probability)
        self.occupied_slots = []
        self.available_slots = n_slots
        self.transmission_probability = transmission_probability

    def trigger(self) -> int:
        for idx in range(self.available_slots):
            if randrange(0, BERNOULLI_TRIES_RESOLUTION) < self.transmission_probability:
                self.occupied_slots += [idx]
        self.announce(VectorMessageBody(self.occupied_slots))
        return len(self.occupied_slots)
