import logging
from typing import Tuple, List, Optional

from tdma_sim_v2.evaluators.distributedConsensusEvaluator import DistributedConsensusEvaluator
from tdma_sim_v2.metrics.experimentDto import AtomicData, AnomalyData, TopologyTestData, TopologyEvolutionTestData
from tdma_sim_v2.model.networks.network import Network
from tdma_sim_v2.utils.buffered_report_writer import BufferedReportWriter


class ExperimentMetrics:
    FILE_ANOMALIES = "result_anomalies_{}_tolerance-{}.tsv"
    FILE_EXACT = "result_full_{}_tolerance-{}.tsv"
    FILE_TOPOLOGY = "result_per_topology_{}_tolerance-{}.tsv"
    FILE_EVOLUTION = "result_per_topology_evolution_{}_tolerance-{}.tsv"

    logger = logging.getLogger(__name__)
    evaluator = DistributedConsensusEvaluator()

    def __init__(self, reports_location: str, strategy: str, failures_tolerance: int):
        self.simulations: List[AtomicData] = []  # extensive list of all recorded simulation results
        self.new_topology_idx = 0
        self.slowest: Optional[Network] = None
        self.fastest: Optional[Network] = None
        self.simulations_report = BufferedReportWriter(reports_location,
                                                       self.FILE_EXACT.format(strategy, failures_tolerance),
                                                       AtomicData.get_header())
        self.anomalies_report = BufferedReportWriter(reports_location,
                                                     self.FILE_ANOMALIES.format(strategy, failures_tolerance),
                                                     AnomalyData.get_header(), buffer_size=1)
        self.topology_report = BufferedReportWriter(reports_location,
                                                    self.FILE_TOPOLOGY.format(strategy, failures_tolerance),
                                                    TopologyTestData.get_header(), buffer_size=1)
        self.topology_evolution_report = BufferedReportWriter(reports_location,
                                                              self.FILE_EVOLUTION.format(strategy, failures_tolerance),
                                                              TopologyEvolutionTestData.get_header(), buffer_size=1)
        self.fault_tolerance = failures_tolerance

    def record_simulation(self, experiment_id: Tuple[int, int, int, int, int, str], network: Network) -> None:
        data = _prepare_data_record(experiment_id, network, self.fault_tolerance)
        self._post_simulation_data(data)

    def record_simulation_details(self, experiment_id: Tuple[int, int, int, int, int, str], network: Network,
                                  valuation: List[float], reason: str, simulation_data: AtomicData = None) -> None:
        data = _prepare_anomaly_record(experiment_id, network, valuation, reason, self.fault_tolerance, simulation_data)
        self._post_anomaly_data(data)

    def record_topology(self) -> None:
        data = TopologyTestData.from_atomic_data_batch(self.simulations[self.new_topology_idx:])
        self.topology_report.write(data)
        self.new_topology_idx = len(self.simulations)

    def record_topologies_batch(self) -> None:
        data = TopologyEvolutionTestData.from_atomic_data_batch(self.simulations)
        self.topology_evolution_report.write(data)
        self.simulations = []
        self.new_topology_idx = 0

    def finalize(self):
        self.simulations_report.close()
        self.anomalies_report.close()
        self.topology_report.close()
        self.topology_evolution_report.close()

    def get_slowest(self) -> Network:
        return self.slowest

    def get_fastest(self) -> Network:
        return self.slowest

    def _post_simulation_data(self, data: AtomicData) -> None:
        self.simulations.append(data)
        self.simulations_report.write(data)

    def _post_anomaly_data(self, data: AnomalyData) -> None:
        self.anomalies_report.write(data)

    def _verify_fastest(self, network: Network) -> None:
        if self.fastest is None:
            self.fastest = network
            return
        n_messages = network.metrics.get_messages_number()
        f_messages = self.fastest.metrics.get_messages_number()
        if n_messages < f_messages:
            self.fastest = network

    def _verify_slowest(self, network: Network) -> None:
        if self.slowest is None:
            self.slowest = network
            return
        n_messages = network.metrics.get_messages_number()
        s_messages = network.metrics.get_messages_number()
        if n_messages > s_messages:
            self.slowest = network


def _prepare_anomaly_record(experiment_id: Tuple[int, int, int, int, int, str], network: Network,
                            valuation: List[float], reason: str, fault_tolerance: int,
                            test_data: AtomicData = None) -> AnomalyData:
    if test_data:
        data = AnomalyData.from_atomic_data_batch(test_data)
    else:
        data = AnomalyData()
        data.set_id(experiment_id)
        data.set_basic_info(network, fault_tolerance)
    data.set_construction_info(network, valuation)
    data.reason = reason
    return data


def _prepare_data_record(experiment_id: Tuple[int, int, int, int, int, str],
                         network: Network, fault_tolerance: int) -> AtomicData:
    data = AtomicData()
    data.set_id(experiment_id)
    data.set_basic_info(network, fault_tolerance)
    return data


def _combine_anomalies(data: AtomicData) -> Optional[str]:
    verdict = []
    if data.incorrect_number:
        verdict += ['converged-to-minority']
    if data.no_common_tilt_number:
        verdict += ['no-common-tilt']
    if data.communication_success_number:
        verdict += ['not-converged']
    if len(verdict):
        return '|'.join(verdict)
    else:
        return None
