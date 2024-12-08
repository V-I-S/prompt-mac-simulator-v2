from functools import reduce
from typing import List, Tuple, Optional

from tdma_sim_v2.model.messages.message import Message
from tdma_sim_v2.model.messages.vectorMessageBody import VectorMessageBody
from tdma_sim_v2.model.networks.network import Network
from tdma_sim_v2.model.nodes.node import Node
from tdma_sim_v2.model.nodes.tSlotsNode import tSlotsNode
from tdma_sim_v2.config.execution import ExecutionConfig


class TdmaNetwork(Network):

    def __init__(self, nodes: List['Node'], n_slots: int):
        super().__init__(nodes)
        self.slots = [0 for _ in range(n_slots)]

    def reset(self, valuation: List[float]) -> None:
        super().reset(valuation)
        self.slots = [0 for _ in range(len(self.slots))]

    def get_valuation(self) -> List[float]:
        # kinds:
        # - no transmission
        # - single transmission (comm. success)
        # - many transmissions (collision)
        return [self._count_slots_with_no_transmissions(),
                self._count_slots_with_single_tranmission(),
                self._count_slots_with_collision()]

    def _count_slots_with_no_transmissions(self):
        return reduce(lambda collected, item: collected + 1 if item == 0 else collected, self.slots, 0)

    def _count_slots_with_single_tranmission(self):
        return reduce(lambda collected, item: collected + 1 if item == 1 else collected, self.slots, 0)

    def _count_slots_with_collision(self):
        return reduce(lambda collected, item: collected + 1 if item > 1 else collected, self.slots, 0)

    def is_communication_success(self, failure_tolerance: int = 0) -> bool:
        failed = 0
        for n in self.nodes:
            if not self.__has_dedicated_channel(n):
                failed += 1
        if ExecutionConfig.failures_tolerance_mode == 'EXACT':
            return failed == failure_tolerance
        elif ExecutionConfig.failures_tolerance_mode == 'LIMIT':
            return failed <= failure_tolerance
        else:
            raise SetupError(f'Unknown failure tolerance mode: {ExecutionConfig.failures_tolerance_mode}')

    def is_communication_success_percentile(self, percentile: float) -> bool:
        jammed_nodes = 0
        for n in self.nodes:
            if not self.__has_dedicated_channel(n):
                jammed_nodes += 1
        return jammed_nodes / len(self.nodes) < (100 - percentile) / 100

    def deliver(self) -> Tuple[Optional['Message'], bool]:
        if not self.relay:
            self.logger.debug('Network relay is empty, sentiment: %.3f', self._count_votes())
            return None, None
        msg = self.relay.pop(0)
        if not isinstance(msg.body, VectorMessageBody):
            self.logger.error('TDMA Network expects slots allocation message from its nodes')
            raise AssertionError('Bad MessageBody Type, expected \'VectorMessageBody\'')

        for s in msg.body.vector:
            self.slots[s] += 1
        return msg, False

    def __has_dedicated_channel(self, node: Node) -> bool:
        assert isinstance(node, tSlotsNode)
        if not node.occupied_slots:
            self.logger.error('Node %s did not occupy any slot!', node)
        for s in node.occupied_slots:
            if self.slots[s] == 1:
                return True
        return False
