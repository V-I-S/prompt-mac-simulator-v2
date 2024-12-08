from typing import List, Generator

from tdma_sim_v2.lab.valuators.valuationGenerator import ValuationGenerator


class ValuationStrategy():
    def __init__(self, valuation_generator_factory: ValuationGenerator):
        self.valuation_generator_factory = valuation_generator_factory
        pass

    def strategy_generator(self, size: int, yields_limit: int = 0) -> Generator[Generator[List[float], None, None], None, None]:
        if size < 0:
            raise AssertionError("Size of the network has to be non-negative")
        if yields_limit < 0:
            raise AssertionError("Limit of generated values has to be non-negative; 0 stands for limitless")

    def get_generator_id(self) -> str:
        raise NotImplementedError
