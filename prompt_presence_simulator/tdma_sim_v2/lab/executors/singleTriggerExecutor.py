import logging
from random import choice
from typing import Optional

from tdma_sim_v2.lab.executors.executor import Executor
from tdma_sim_v2.model.messages.message import Message
from tdma_sim_v2.model.networks.network import Network


class SingleTriggerExecutor(Executor):
    logger = logging.getLogger(__name__)

    def __init__(self, network: Network):
        super().__init__(network)

    def run(self, steps_limit: int = 0) -> None:
        self.run_validate_input(steps_limit)
        self._trigger_rand_node()
        delivered, is_converged = self.network.deliver()
        steps = 1
        while not _finished_processing(is_converged, delivered, steps, steps_limit):
            delivered, is_converged = self.network.deliver()
            steps += 1

    def _trigger_rand_node(self) -> None:
        node = choice(self.network.nodes)
        self.logger.debug('%s triggering', node)
        node.trigger()


def _finished_processing(network_converged: bool, delivered: Optional[Message], steps: int, steps_limit: int):
    return network_converged or not delivered or (steps >= steps_limit > 0)
