from tdma_sim_v2.model.networks.network import Network


class Executor:
    def __init__(self, network: Network):
        self.network = network

    def run(self, steps_limit: int = 0) -> None:
        """
        Define path of execution - like trigger random nodes until convergence.
        Intakes parameter defining the upper step number limit, while zero is intepreted as no-limit.
        """
        raise NotImplementedError

    @staticmethod
    def run_validate_input(steps_limit: int):
        if steps_limit < 0:
            raise AssertionError("Steps' limit has to be non-negative. 0 stands for no limit")
