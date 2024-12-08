from typing import List, Generator

from tdma_sim_v2.lab.valuators.valuationGenerator import ValuationGenerator
from tdma_sim_v2.lab.valuestrat.valuationStrategy import ValuationStrategy


class AllRatiosYesNoValuationStrategy(ValuationStrategy):
    def __init__(self, generator_factory: ValuationGenerator):
        super().__init__(generator_factory)
        self.generator_id = ''

    def strategy_generator(self, size: int, yields_limit: int = 0) -> Generator[Generator[List[float], None, None], None, None]:
        super().strategy_generator(size, yields_limit)
        for yes_num in range(size):
            self.generator_id = f'{yes_num}:{size - yes_num}'
            yield self.valuation_generator_factory.generator(size, yes_num, size - yes_num, yields_limit)

    def get_generator_id(self) -> str:
        return self.generator_id
