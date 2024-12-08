import concurrent.futures
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tdma_calc.api import tslots_success_probability, bernoulli_success_probability

from tdma_sim_v2.utils.exceptions.sanityCheckError import SanityCheckError


class Summary:
    FILE_FIGURE = 'result_visualization.png'
    FILE_CALCULATION_RESULTS = 'result_calculation_data_10.tsv'
    FILE_PARTIAL_CALCULATION_RESULTS = 'partial_calculation_data.tsv'

    logger = logging.getLogger(__name__)

    def __init__(self, config: dict):
        self.experiment_name = config['experiment.name']
        self.experiment_dir = config['experiment.dir']
        self.n_min_transmitters = config['experiment.size.start']
        self.n_max_transmitters = config['experiment.size.stop']
        self.n_slots = config['experiment.available-slots']
        self.n_max_sapling = config['experiment.strategy.evolution.stop']
        self.execution_cores = config.get('experiment.execution-cores', 1)

    def sanity_check(self):
        if self.n_min_transmitters != self.n_max_transmitters:
            raise SanityCheckError('Plot configuration operates exclusively on the starting number of transmitters')

    def plot(self):
        if 'bernoulli' in self.experiment_name:
            rslt = self._get_bernoulli_experiment_results()
            self._plot_bernoulli(rslt)
        else:
            rslt = self._get_experiment_results()
            self._plot_selective(rslt)

    def _plot_bernoulli(self, experiment: Dict[str, List[float]]):
        sampling = np.arange(0., 1., 0.005)
        probability_success = [bernoulli_success_probability(self.n_min_transmitters, self.n_slots, p) for p in sampling]
        self._store_calculation_to_file('Bernoulli', list(sampling), probability_success)
        fig, ax = plt.subplots()
        ax.plot(sampling, probability_success, label='Model calculation')
        # ax.plot(experiment['index'], experiment['value'], 'r.', label='Experiment results')
        ax.set_xlabel('Agent\'s transmission probability in the single slot (p)')
        ax.set_ylabel('Communication success probability')
        # ax.legend(loc='upper right')
        plt.axis([0, 1, 0, 1])
        plt.title('Bernoulli-Trials Strategy')
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_FIGURE))
        plt.show()

    def _plot_selective(self, experiment: Dict[str, List[float]]):
        sampling = np.arange(0, self.n_max_sapling + 1, 1)
        params = [[self.n_min_transmitters, self.n_slots, s] for s in sampling]
        print('Obtaining selective method calculation results...')
        if self.is_calculated_already('result_calculation_data_10.tsv'):
            probability_success = self._get_values('result_calculation_data_10.tsv')['value']
        else:
            probability_success = self._run_in_parallel(params)
            self._store_calculation_to_file('Selective', list(sampling), probability_success)
        print('Creating plot out of experimental and calculation data...')
        fig, ax = plt.subplots()
        ax.plot(sampling, probability_success, label='Model calculation')
        # ax.plot(experiment['index'], experiment['value'], 'r.', label='Experiment results')
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel('Cardinality of agent\'s transmission set (t)')
        ax.set_ylabel('Communication success probability')
        # ax.legend(loc='upper right')
        plt.axis([0, self.n_slots, 0, 1])
        plt.title('t-Slots Strategy')
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_FIGURE))
        plt.show()

    def _run_in_parallel(self, argument_groups: List) -> List[float]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.execution_cores) as executor:
            pool = executor.map(self._calculate_selective_with_output_redirection, argument_groups)
            return list(pool)

    def _calculate_selective_with_output_redirection(self, args) -> float:
        calculation_result = self._retrieve_cached_calculation(args[2])
        if not calculation_result:
            print(f'No previous calculation found. Calculating for {args[2]} selections...')
            output_file = 'tdma_summary_' + datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + '_pid_' + str(os.getpid()) + '.out'
            sys.stdout = open(os.path.join(self.experiment_dir, output_file), "w", buffering=1)
            sys.stdout.write(f'Starting process: {os.getpid()}\n')
            calculation_result = tslots_success_probability(*args)
            self._store_partial_calculation(args[2], calculation_result)
        else:
            print(f'Previous calculation found: {args[2]}\t{calculation_result}')
        return calculation_result

    def _store_calculation_to_file(self, name: str, steps: List[float], results: List[float]) -> None:
        with open(os.path.join(self.experiment_dir, self.FILE_CALCULATION_RESULTS), 'a') as output:
            output.write(f'{name} results (calculated {datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}):\n')
            for record in zip(steps, results):
                output.write('{:0.4f}\t{:0.6f}\n'.format(*record))

    def _get_bernoulli_experiment_results(self) -> Dict[str, List[float]]:
        results_file = os.path.join(self.experiment_dir, 'result_per_topology_evolution_v1.tsv')
        df = pd.read_csv(results_file, delimiter='\t')
        df['step'] = df['step'].apply(lambda v: v / 100.0)
        df['#-comm-success'] = df.apply(lambda v: float(v['#-comm-success']) / v['#-of-tests'], axis=1)
        prob_index = df['step'].tolist()
        prob_values = df['#-comm-success'].tolist()
        return {"index": prob_index, "value": prob_values}

    def _get_experiment_results(self) -> Dict[str, List[float]]:
        results_file = os.path.join(self.experiment_dir, 'result_per_topology_evolution_v1.tsv')
        df = pd.read_csv(results_file, delimiter='\t')
        df['#-comm-success'] = df.apply(lambda v: float(v['#-comm-success']) / v['#-of-tests'], axis=1)
        prob_index = df['step'].tolist()
        prob_values = df['#-comm-success'].tolist()
        return {"index": prob_index, "value": prob_values}

    def is_calculated_already(self, file_name: str):
        return os.path.isfile(os.path.join(self.experiment_dir, file_name))

    def _get_values(self, file_name: str, scaler: int = 1) -> Dict[str, List[float]]:
        results_file = os.path.join(self.experiment_dir, file_name)
        df = pd.read_csv(results_file, delimiter='\t', header=None, skiprows=1)
        print(df[0])
        prob_index = df[0].apply(lambda v: v*scaler).tolist()
        prob_values = df[1].tolist()
        return {"index": prob_index, "value": prob_values}

    def _store_partial_calculation(self, index: float, calculation_result: float):
        file_path = os.path.join(self.experiment_dir, self.FILE_PARTIAL_CALCULATION_RESULTS)
        with open(file_path, 'a') as partial_results_file:
            print(f'Storing partial calculation results in {self.FILE_PARTIAL_CALCULATION_RESULTS}: {index} {calculation_result}')
            partial_results_file.write(f'{index:0.4f}\t{calculation_result:0.6f}\n')
            partial_results_file.flush()

    @staticmethod
    def _transpose_file(file_name: str):
        df = pd.read_csv(file_name, delimiter='\t', header=None, skiprows=1).transpose()
        df.to_csv(file_name + '_transposed.tsv', sep='\t', header=False, index=False)

    def _retrieve_cached_calculation(self, index: float) -> Optional[float]:
        results_file = os.path.join(self.experiment_dir, self.FILE_PARTIAL_CALCULATION_RESULTS)
        df = pd.read_csv(results_file, delimiter='\t', header=None, skiprows=1)
        if df.loc[df[0] == index].empty:
            return None
        return float(df.loc[df[0] == index][1])
