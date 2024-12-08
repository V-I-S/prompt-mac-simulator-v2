from typing import List, Generator

from tdma_sim_v2.lab.valuators.valuationGenerator import ValuationGenerator


class AllPermutationsValuationGenerator(ValuationGenerator):
    def __init__(self):
        super().__init__()

    def generator(self, size: int, yes_voters: int, no_voters: int, yields_limit: int = 0) -> Generator[List[float], None, None]:
        super().generator(size, yes_voters, no_voters, yields_limit)
        if yes_voters + no_voters != size:
            raise NotImplementedError("Initially neutral nodes not supported")
        for p in range(0, 2 ** size):
            candidate = list(f'{p:b}'.zfill(size))
            candidate = list(map(lambda x: int(x), candidate))
            if sum(candidate) == yes_voters:
                yield [-1.0 if c == 0 else 1.0 for c in candidate]
