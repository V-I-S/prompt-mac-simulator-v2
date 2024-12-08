from typing import List, Generator


class ValuationGenerator:
    def __init__(self):
        pass

    def generator(self, size: int, yes_voters: int, no_voters: int, yields_limit: int = 0) -> Generator[List[float], None, None]:
        """
        Usage: generator(<network_size>, <y>, <n>, <max_number_of_yelds>), where <y> + <n> <= <network_size>.
        If <y> + <n> < <network_size>, the surplus nodes are valuated to 0 (careful - in some interpretations it may be interpreted different than 'hesitating').
        """
        if yes_voters + no_voters > size:
            raise AssertionError("Sum of valuated nodes cannot exceed total number of nodes")
        if yields_limit < 0:
            raise AssertionError("Limit of generated values has to be non-negative; 0 stands for limitless")
