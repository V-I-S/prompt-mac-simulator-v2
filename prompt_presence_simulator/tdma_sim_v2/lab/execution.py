import re
from datetime import datetime
from typing import List, Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

from numpy import arange, sqrt
from prompt_mac_calculator.api import fault_tolerant_bernoulli_success_probability, \
    fault_tolerant_tslots_success_probability

from tdma_sim_v2.config.execution import BERNOULLI_TRIES_RESOLUTION, ExecutionConfig
from tdma_sim_v2.lab.experiment import Experiment
from tdma_sim_v2.utils.exceptions.sanityCheckError import SanityCheckError


class Execution:
    FILE_FIGURE = 'result_visualization.png'
    RESULT_FILE_FORMAT: str = 'result_per_topology_evolution_{}_tolerance-{}.tsv'
    CALCULATION_FILE_FORMAT = 'result_calculation_data_{}_tolerance-{}.tsv'
    ANOMALIES_FILE_FORMAT = 'result_anomalies_tSlotsNode_tolerance-0.tsv'

    def __init__(self, config: dict):
        self.config = config
        self.experiment_dir = config['experiment.dir']
        self.n_slots = config['experiment.available-slots']
        self.n_min_transmitters = config['experiment.size.start']
        self.n_max_transmitters = config['experiment.size.stop'] + 1
        self.n_transmitters_step = config['experiment.size.step']
        self.failures_acceptance_rng = range(config['experiment.failures-tolerance.start'],
                                             config['experiment.failures-tolerance.stop'] + 1,
                                             config['experiment.failures-tolerance.step'])
        self.strategy_evolution_rng: range = range(config['experiment.strategy.evolution.start'],
                                                   config['experiment.strategy.evolution.stop'] + 1,
                                                   max(1, config['experiment.strategy.evolution.step']))
        self.experiment_model = self.config['experiment.model.node']
        self.experiment_file_regex = re.compile(r'result_per_topology_evolution_(\w+)_tolerance-(\d+).tsv')
        self.calculation_file_regex = re.compile(r'result_calculation_data_(\w+)_tolerance-(\d+).tsv')

    def sanity_check(self) -> None:
        """Executes verifications of unsanitized configuration to fail fast.
        Passes silently if all is fine, throws SanityCheckError otherwise."""
        if self.size_rng.n_min_transmitters != self.n_max_transmitters:
            raise SanityCheckError('Network size variation not allowed here', 'experiment.size')

    def perform(self) -> None:
        exp_results = self._ensure_all_experiments_done()
        # calc_results = self._ensure_all_calculations_done()
        # self._plot_experiments(exp_results, calc_results)
        # extras
        # Partitions(self.config).plot()  # for load statistics of each group
        from tdma_sim_v2.lab.extras.confirmation import Confirmation  # circular dependency :(
        Confirmation(self, self.config).plot()  # automated comparison with calculation results
        # from tdma_sim_v2.lab.extras.comparison import Comparison  # circular dependency :(
        # Comparison(self, self.config).plot()  # for the extra comparison with other strategy

    def get_experiment_results(self, file: str, model: str, complete_file: str = None) -> Dict[str, List[float]]:
        results_file = os.path.join(self.experiment_dir, file)
        complete_file = os.path.join(self.experiment_dir, complete_file) if type(complete_file) == str else None
        df = pd.read_csv(results_file, delimiter='\t')
        if model == 'BtNode':
            df['step'] = df['step'].apply(lambda v: v / BERNOULLI_TRIES_RESOLUTION)
        df['#-comm-success'] = df.apply(lambda v: float(v['#-comm-success']) / v['#-of-tests'], axis=1)
        prob_index = df['step'].tolist()
        prob_values = df['#-comm-success'].tolist()
        if complete_file is not None:
            std_err = self._collect_standard_errors(complete_file)
        else:
            std_err = [0.0] * len(prob_index)
        return {"index": prob_index, "value": prob_values, "stderr": std_err}

    def _collect_standard_errors(self, complete_file: str) -> List[float]:
        df = pd.read_csv(complete_file, delimiter='\t')
        return [self._calculate_standard_error(df, step) for step in df['step'].unique()]

    @staticmethod
    def _calculate_standard_error(df: pd.DataFrame, step: int) -> float:
        results_for_step = df[df['step'] == step]['#-comm-success']
        prob = results_for_step.sum() / len(results_for_step)
        std_dev = sqrt(prob - prob ** 2)
        return 3.290526731 * sqrt(std_dev ** 2 / len(results_for_step))

    def get_calculation_results(self, file: str) -> Dict[str, List[float]]:
        results_file = os.path.join(self.experiment_dir, file)
        df = pd.read_csv(results_file, delimiter='\t', skiprows=1, index_col=False)
        index = df['index'].tolist()
        prob_values = df['success_probability'].tolist()
        return {"index": index, "value": prob_values}

    def _ensure_all_experiments_done(self):
        results = list()
        for ft_step in self.failures_acceptance_rng:
            results.append(self._ensure_experiment_done(ft_step))
        return results

    def _ensure_experiment_done(self, failures_tolerance: int) -> str:
        print(f'Performing experiments for failure_tolerance={failures_tolerance}...')
        file = self.RESULT_FILE_FORMAT.format(self.experiment_model, failures_tolerance)
        if os.path.isfile(os.path.join(self.experiment_dir, file)):
            print(f'Experiment results already collected for failure_tolerance={failures_tolerance}')
            return file
        Experiment(self.config, failures_tolerance).perform()
        return file

    def _ensure_all_calculations_done(self):
        results = list()
        for ft_step in self.failures_acceptance_rng:
            results.append(self._ensure_calculation_done(ft_step))
        return results

    def _ensure_calculation_done(self, failures_tolerance: int) -> str:
        print(f'Performing calculations for k={self.n_min_transmitters}, failures tolerance={failures_tolerance}...')
        file = self.CALCULATION_FILE_FORMAT.format(self.experiment_model, failures_tolerance)
        if os.path.isfile(os.path.join(self.experiment_dir, file)):
            print(f'Calculation results already collected for failure_tolerance={failures_tolerance}')
            return file
        return self._perform_calculation(file, failures_tolerance)

    def _perform_calculation(self, file: str, failures_tolerance: int) -> str:
        sampling, values = None, None
        if self.experiment_model == 'BtNode':
            sampling, values = self._calculate_bt(failures_tolerance)
        elif self.experiment_model == 'tSlotsNode':
            sampling, values = self._calculate_tslots(failures_tolerance)
        else:
            print('Error: unrecognized agents'' strategy')
        self._store_to_file(file, failures_tolerance, sampling, values)
        return file

    def _store_to_file(self, file: str, failures_tolerance: int, sampling: List[float], values: List[float]) -> None:
        with open(os.path.join(self.experiment_dir, file), 'w') as output:
            output.write(f'{self.experiment_model} results, failures tolerance={failures_tolerance} '
                         f'(calculated {datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}):\n')
            output.write('index\tsuccess_probability\n')
            for record in zip(sampling, values):
                output.write('{:0.6f}\t{:0.8f}\n'.format(*record))

    def _plot_experiments(self, experiments: List[str], calculations: List[str]) -> None:
        """Any model derivations in plot mechanism comes from the file naming, so the comparisons may be easily made."""
        print(f'Summarizing results in the plot...')
        ax = self. _prepare_plot()
        for file in experiments:
            model = self.experiment_file_regex.match(file).group(1)
            tolerance = int(self.experiment_file_regex.match(file).group(2))
            data = self.get_experiment_results(file, model)
            self._add_experiment_data_to_plot(ax, data, tolerance)
        for file in calculations:
            tolerance = int(self.calculation_file_regex.match(file).group(2))
            data = self.get_calculation_results(file)
            self._add_calculation_data_to_plot(ax, data, tolerance)
        self._finalize_plot()

    def _calculate_bt(self, failures_tolerance: int) -> Tuple[List[float], List[float]]:
        sampling = np.arange(0., 1., 0.002)
        probability_success = []
        iterations = 0
        for agents in range(self.n_min_transmitters, self.n_max_transmitters, self.n_transmitters_step):
            print(f'Adding calculated results for k={agents}, failures tolerance={failures_tolerance}')
            probability_success += [
                fault_tolerant_bernoulli_success_probability(agents, self.n_slots, p, failures_tolerance)
                for p in sampling]
            iterations += 1
        return list(sampling) * iterations, probability_success

    def _calculate_tslots(self, failures_tolerance: int) -> Tuple[List[float], List[float]]:
        sampling = np.arange(0, self.strategy_evolution_rng.stop, 1)
        probability_success = []
        iterations = 0
        for agents in range(self.n_min_transmitters, self.n_max_transmitters, self.n_transmitters_step):
            print(f'Adding calculated results for k={agents}, failures tolerance={failures_tolerance}')
            probability_success += [
                fault_tolerant_tslots_success_probability(agents, self.n_slots, p, failures_tolerance)
                for p in sampling]
            iterations += 1
        return list(sampling) * iterations, probability_success

    def _prepare_plot(self):
        fig, ax = plt.subplots()
        # ax.plot(sampling, probability_success, label='Model calculation')
        ax.set_xlabel('Agent\'s transmission probability in the single slot (p)')
        ax.set_ylabel('Communication success probability')
        ax.legend(loc='upper right')
        return ax

    @staticmethod
    def _add_experiment_data_to_plot(ax, experiment: Dict[str, List[float]], failures_tolerance: int) -> None:
        print(f'Plotting experiment results for failures tolerance={failures_tolerance}...')
        ax.plot(experiment['index'], experiment['value'], 'r.',
                label=f'Experiment results - {failures_tolerance} failed node(s) accepted')

    @staticmethod
    def _add_calculation_data_to_plot(ax, calculation: Dict[str, List[float]], failures_tolerance: int) -> None:
        print(f'Plotting calculation results for failures tolerance={failures_tolerance}...')
        ax.plot(calculation['index'], calculation['value'],
                label=f'Model calculation - {failures_tolerance} failed node(s) accepted')

    def _finalize_plot(self):
        if self.experiment_model == 'BtNode':
            plt.axis([0, 1, 0, 1])
        elif self.experiment_model == 'tSlotsNode':
            plt.axis([0, self.strategy_evolution_rng.stop, 0, 1])
        else:
            print('Error: Plotting issue, unrecognized agents'' strategy')
        plt.title('Bernoulli-Trials Strategy')
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_FIGURE))
        plt.show()
