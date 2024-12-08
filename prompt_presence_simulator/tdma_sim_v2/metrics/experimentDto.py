from __future__ import annotations

import logging
from statistics import mean
from typing import List, Tuple, Any

from tdma_sim_v2.evaluators.distributedConsensusEvaluator import DistributedConsensusEvaluator
from tdma_sim_v2.model.networks.network import Network
from tdma_sim_v2.utils.buffered_report_writer import BufferedReportWriter

VALUE_SEPARATOR = BufferedReportWriter.DATA_SEPARATOR


# TODO Re-consider design: composition instead of inheritance
class TopologyEvolutionTestData:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.slots: int = 0
        self.network_size: int = 0
        self.topology_step: int = 0
        self.tests_number: int = 0
        self.communication_success_number: int = 0
        self.success_99_percentile: int = 0
        self.avg_message_number: int = 0
        self.avg_deliveries_number: int = 0

    def __str__(self):
        ordered_fields = [self.slots, self.network_size, self.topology_step, self.tests_number, self.communication_success_number,
                          self.success_99_percentile, self.avg_message_number, self.avg_deliveries_number]
        return VALUE_SEPARATOR.join(map(_str_format, ordered_fields))

    def _return_when_assembled_fields(self, atomics: List[AtomicData]) -> TopologyEvolutionTestData:
        self.network_size = _extract_network_size(atomics)
        self.topology_step = _extract_topology_step(atomics)
        self.tests_number = _summarize_tests_number(atomics)
        self.communication_success_number = _summarize_communication_success_number(atomics)
        self.success_99_percentile = _summarize_success_99_percentile_success_number(atomics)
        self.avg_message_number = _summarize_avg_message_number(atomics)
        self.avg_deliveries_number = _summarize_avg_deliveries_number(atomics)
        return self

    @classmethod
    def from_atomic_data_batch(cls, atomics: List[AtomicData]) -> TopologyEvolutionTestData:
        try:
            return cls()._return_when_assembled_fields(atomics)
        except AssertionError as err:
            cls.logger.error('Test data construction failed due to inconsitency: %s', err)
            return cls()

    @staticmethod
    def get_header() -> List[str]:
        return ['slots', 'network-size', 'step', '#-of-tests', '#-comm-success', '#-99-percentile', 'avg-#-msg', 'avg-deliveries-#']


class TopologyTestData(TopologyEvolutionTestData):
    def __init__(self):
        super().__init__()
        self.topology_id: int = 0

    def __str__(self):
        ordered_fields = [self.slots, self.network_size, self.topology_step, self.topology_id, self.tests_number,
                          self.communication_success_number, self.avg_message_number, self.avg_deliveries_number]
        return VALUE_SEPARATOR.join(map(_str_format, ordered_fields))

    def _return_when_assembled_fields(self, atomics: List[AtomicData]) -> TopologyTestData:
        super()._return_when_assembled_fields(atomics)
        self.topology_id = _extract_topology_id(atomics)
        return self

    @classmethod
    def from_atomic_data_batch(cls, atomics: List[AtomicData]) -> TopologyTestData:
        try:
            return cls()._return_when_assembled_fields(atomics)
        except AssertionError as err:
            cls.logger.error('Test data construction failed due to inconsitency: %s', err)
            return cls()
        # my support reminder // return Ela_kocha
        # little Xawery's comment preserved // q2§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§xsxz

    @staticmethod
    def get_header() -> List[str]:
        return ['slots', 'network-size', 'step', 'experiment-id', '#-of-tests', '#-comm-success', 'avg-#-msg', 'avg-deliveries-#']


# todo: could be immutable
class AtomicData(TopologyTestData):
    evaluator = DistributedConsensusEvaluator()

    def __init__(self):
        super().__init__()
        self.iteration: int = 0
        self.valuation_strat_id: str = ''

    def __str__(self):
        ordered_fields = [self.iteration, self.slots,  self.network_size, self.topology_step, self.topology_id, self.valuation_strat_id, self.tests_number,
                          self.communication_success_number, self.success_99_percentile, self.avg_message_number, self.avg_deliveries_number]
        return VALUE_SEPARATOR.join(map(_str_format, ordered_fields))

    def set_id(self, experiment_id: Tuple[int, int, int, int, int, str]) -> None:
        self.iteration, self.slots, self.network_size, self.topology_step, self.topology_id, self.valuation_strat_id = experiment_id

    def set_basic_info(self, network: Network, failure_tolerance: int) -> None:
        self.tests_number = 1
        self.communication_success_number = int(self.evaluator.check_communication_success(network, failure_tolerance))
        self.success_99_percentile = int(self.evaluator.check_communication_success_percentile(network, 99.0))
        self.avg_message_number = network.metrics.get_messages_number()
        self.avg_deliveries_number = network.metrics.get_deliveries_number()

    @staticmethod
    def get_header():
        return ['iteration', 'slots', 'network-size', 'step', 'experiment-id', 'valuation-strategy', '#-of-tests',
                '#-comm-success', '#-99-percentile', 'avg-#-msg', 'avg-deliveries-#']


class AnomalyData(AtomicData):
    def __init__(self):
        super().__init__()
        self.topology: List[List[int]] = []
        self.init_valuation: List[float] = []
        self.end_valuation: List[float] = []
        self.reason: str = ''

    def __str__(self):
        ordered_fields = [self.iteration, self.slots, self.network_size, self.topology_step, self.topology_id, self.topology, self.valuation_strat_id,
                          self.init_valuation, self.end_valuation, self.tests_number, self.communication_success_number,
                          self.avg_message_number, self.avg_deliveries_number, self.reason]
        return VALUE_SEPARATOR.join(map(_str_format, ordered_fields))

    def set_construction_info(self, network: Network, valuation: List[float]):
        self.topology = network.get_topology_repr()
        self.init_valuation = valuation
        self.end_valuation = network.get_valuation()

    @classmethod
    def from_atomic_data_batch(cls, data: AtomicData) -> AnomalyData:
        inst = cls()
        inst.set_id((data.iteration, data.slots, data.network_size, data.topology_step, data.topology_id, data.valuation_strat_id))
        inst.tests_number = data.tests_number
        inst.communication_success_number = data.communication_success_number
        inst.avg_message_number = data.avg_message_number
        inst.avg_deliveries_number = data.avg_deliveries_number
        return inst

    @staticmethod
    def get_header():
        return ['iteration', 'slots', 'network-size', 'step', 'experiment-id', 'topology', 'valuation-strategy', 'start-picture',
                'end-picture', '#-of-tests', '#-comm-success', 'avg-#-msg', 'avg-deliveries-#', 'reason']


def _extract_network_size(atomics: List[AtomicData]) -> int:
    min_size = min(map(lambda a: a.network_size, atomics))
    max_size = max(map(lambda a: a.network_size, atomics))
    if min_size != max_size:
        raise AssertionError(f'Network size inconsistent, shall be static while: min={min_size}, max={max_size}')
    return min_size


def _extract_topology_step(atomics: List[AtomicData]) -> int:
    min_step = min(map(lambda a: a.topology_step, atomics))
    max_step = max(map(lambda a: a.topology_step, atomics))
    if min_step != max_step:
        raise AssertionError(f'Topology step inconsistent, shall be static while: min={min_step}, max={max_step}')
    return min_step


def _extract_topology_id(atomics: List[AtomicData]) -> int:
    min_id = min(map(lambda a: a.topology_id, atomics))
    max_id = max(map(lambda a: a.topology_id, atomics))
    if min_id != max_id:
        raise AssertionError(f'Topology id inconsistent, shall be static while: min={min_id}, max={max_id}')
    return min_id


def _summarize_tests_number(atomics: List[AtomicData]) -> int:
    return sum(map(lambda a: a.tests_number, atomics))


def _summarize_incorrect_number(atomics: List[AtomicData]) -> int:
    return sum(map(lambda a: a.incorrect_number, atomics))


def _summarize_communication_success_number(atomics: List[AtomicData]) -> int:
    return sum(map(lambda a: a.communication_success_number, atomics))


def _summarize_success_99_percentile_success_number(atomics: List[AtomicData]) -> int:
    return sum(map(lambda a: a.success_99_percentile, atomics))


def _summarize_no_common_tilt_number(atomics: List[AtomicData]) -> int:
    return sum(map(lambda a: a.no_common_tilt_number, atomics))


def _summarize_avg_message_number(atomics: List[AtomicData]) -> int:
    return mean(map(lambda a: a.avg_message_number, atomics))


def _summarize_avg_deliveries_number(atomics: List[AtomicData]) -> int:
    return mean(map(lambda a: a.avg_deliveries_number, atomics))


def _str_format(field: Any) -> str:
    return f'{field:.4f}' if isinstance(field, float) else str(field)
