import logging
import random
from typing import Type, Generator, List, Tuple

from tdma_sim_v2.lab.executors.executor import Executor
from tdma_sim_v2.metrics.experimentMetrics import ExperimentMetrics
from tdma_sim_v2.model.networks.network import Network
from tdma_sim_v2.model.networks.tdmaNetwork import TdmaNetwork
from tdma_sim_v2.model.nodes.node import Node
from tdma_sim_v2.lab.topologies.topologyGenerator import TopologyGenerator
from tdma_sim_v2.model.nodes.tSlotsNode import tSlotsNode
from tdma_sim_v2.utils.exceptions.sanityCheckError import SanityCheckError
from tdma_sim_v2.utils.importUtils import *
from tdma_sim_v2.utils.progressInsight import ProgressInsight
from tdma_sim_v2.lab.valuators.valuationGenerator import ValuationGenerator
from tdma_sim_v2.lab.valuestrat.valuationStrategy import ValuationStrategy


class Experiment:
    MODULE_NETWORKS = 'tdma_sim_v2.model.networks'
    MODULE_NODES = 'tdma_sim_v2.model.nodes'
    MODULE_EXECUTORS = 'tdma_sim_v2.lab.executors'
    MODULE_TOPOLOGIES = 'tdma_sim_v2.lab.topologies'
    MODULE_VALUATORS = 'tdma_sim_v2.lab.valuators'
    MODULE_VALUATION_STRATEGIES = 'tdma_sim_v2.lab.valuestrat'

    logger = logging.getLogger(__name__)

    def __init__(self, config: dict, failures_tolerance: int = 0):
        self.node_type: Type[tSlotsNode] = class_by_name(self.MODULE_NODES, config['experiment.model.node'])
        self.network_type: Type[TdmaNetwork] = class_by_name(self.MODULE_NETWORKS, config['experiment.model.network'])
        self.executor_type: Type[Executor] = class_by_name(self.MODULE_EXECUTORS, config['experiment.model.executor'])

        self.size_rng: range = range(config['experiment.size.start'],
                                     config['experiment.size.stop'] + 1,
                                     max(1, config['experiment.size.step']))
        self.slots_num: int = config['experiment.available-slots']
        self.strategy_evolution_rng: range = range(config['experiment.strategy.evolution.start'],
                                                   config['experiment.strategy.evolution.stop'] + 1,
                                                   max(1, config['experiment.strategy.evolution.step']))
        self.trials_per_strategy: range = range(config['experiment.strategy.trials-per-strategy'])

        self.valuation_generator: ValuationGenerator = instantiate(self.MODULE_VALUATORS, config['experiment.valuation.generator'])
        self.valuation_strategy: ValuationStrategy = instantiate(self.MODULE_VALUATION_STRATEGIES, config['experiment.valuation.strategy'],
                                                                 self.valuation_generator)
        self.max_valuations_per_topology: int = config['experiment.valuation.max-instances-per-topology']
        self.max_steps_per_valuation: int = config['experiment.valuation.max-steps-per-instance']

        self.store_every_nth_test_metrics: int = config['experiment.sampling-frequency']

        random.seed(config['experiment.random-seed'])
        self.banner = config['experiment.header']
        self.description = config['experiment.description']
        self.metrics = ExperimentMetrics(config['experiment.dir'], config['experiment.model.node'], failures_tolerance)
        self.insight = ProgressInsight(self.logger)
        self.iteration = 0

    def sanity_check(self) -> None:
        """Executes verifications of unsanitized configuration to fail fast. Passes silently if all is fine, throws SanityCheckError otherwise."""
        if (self.size_rng.stop - self.size_rng.start) * self.size_rng.step < 0:
            raise SanityCheckError('Size range does not represent valid domain of sizes', 'experiment.size')
        if (self.strategy_evolution_rng.stop - self.strategy_evolution_rng.start) * self.strategy_evolution_rng.step < 0:
            raise SanityCheckError('Topology evolution range is not well-defined', 'experiment.topology.evolution')
        if self.trials_per_strategy.stop < 0:
            raise SanityCheckError('Number of topologies per evolution has to be positive', 'experiment.topology.instances-per-evolution')
        if self.max_valuations_per_topology < 0:
            raise SanityCheckError('Defined valuations per topology limit has to be non-negative. 0 turns off the limit', 'experiment.valuation.max-instances-per-topology')
        if self.max_steps_per_valuation < 0:
            raise SanityCheckError('Defined experiment steps per valuation limit has to be non-negative. 0 turns off the limit', 'experiment.valuation.max_steps_per_valuation')

    def perform(self):
        self.insight.intro(self.banner, self.description)
        for size in self.size_rng:
            valuation = [0] * size
            for strategy_step in self.strategy_evolution_rng:
                self.insight.phase_begin(size, strategy_step, self.iteration)
                for trial in self.trials_per_strategy:
                    self.iteration += 1
                    self.insight.progress(self.iteration)
                    trial_id_human = trial + 1
                    network = self._prepare_network(size, self.slots_num, strategy_step)
                    experiment_id = (self.iteration, self.slots_num, size, strategy_step, trial_id_human, '-')
                    self._execute_test(network)
                    self._record_test(experiment_id, network, valuation)
                self.metrics.record_topologies_batch()
                self.insight.phase_end()
        self.insight.finalizing(self.banner, self.iteration)
        self.metrics.finalize()
        self.insight.finished(self.banner)

    def _prepare_network(self, network_size: int, slots_num: int, strategy_step: int) -> Network:
        nodes = [self.node_type(idx, slots_num, strategy_step) for idx in range(network_size)]
        return self.network_type(nodes, slots_num)

    def _execute_test(self, network: Network) -> None:
        self.executor_type(network).run(self.max_steps_per_valuation)

    def _record_test(self, experiment_id: Tuple[int, int, int, int, int, str], network: Network, valuation: List[float]) -> None:
        self.metrics.record_simulation(experiment_id, network)
        if self._shall_report_detailed_metrics():
            self.metrics.record_simulation_details(experiment_id, network, valuation, 'sampling')

    def _shall_report_detailed_metrics(self):
        return self.store_every_nth_test_metrics > 0 and self.iteration % self.store_every_nth_test_metrics == 0
