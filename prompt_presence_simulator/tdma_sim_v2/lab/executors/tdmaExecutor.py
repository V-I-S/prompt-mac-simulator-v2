import logging
from random import choice
from typing import Optional

from tdma_sim_v2.lab.executors.executor import Executor
from tdma_sim_v2.model.messages.message import Message
from tdma_sim_v2.model.networks.network import Network


class TdmaExecutor(Executor):
    logger = logging.getLogger(__name__)

    def __init__(self, network: Network):
        super().__init__(network)

    def run(self, steps_limit: int = 0) -> None:
        self.run_validate_input(steps_limit)
        for n in self.network.nodes:
            n.trigger()
            self.network.deliver()


def _finished_processing(network_converged: bool, delivered: Optional[Message], steps: int, steps_limit: int):
    return network_converged or not delivered or (steps >= steps_limit > 0)
