import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import binomial
from tdma_calc.api import tslots_success_probability, bernoulli_success_probability

from tdma_sim_v2.utils.exceptions.sanityCheckError import SanityCheckError


class Custom:
    FILE_FIGURE = 'results_comparison.png'
    FILE_TABLE = 'results_comparison.tsv'
    FILE_SINGLE_FIGURE = 'results.png'
    FILE_SINGLE_TABLE = 'results.tsv'

    logger = logging.getLogger(__name__)

    def __init__(self, config: dict):
        self.experiment_name = config['experiment.name']
        self.experiment_dir = config['experiment.dir']
        self.n_min_transmitters = config['experiment.size.start']
        self.n_max_transmitters = config['experiment.size.stop']
        self.n_slots = config['experiment.available-slots']
        self.execution_cores = config.get('experiment.execution-cores', 1)
        self.test_range = self.n_slots // self.n_min_transmitters + 4

    def sanity_check(self):
        if self.n_min_transmitters != self.n_max_transmitters:
            raise SanityCheckError('Plot configuration operates exclusively on the starting number of transmitters')

    def compare_2_transmitters(self):
        sampling = np.arange(0, self.n_slots, 1)
        selective = [tslots_success_probability(2, self.n_slots, s) for s in sampling]
        simple = [self._selective_simplified_formula(self.n_slots, s) for s in sampling]
        self._store_results_comparison_to_file(sampling, selective, simple)
        self._plot_results_comparison(sampling, selective, simple)

    def test_transmitters_for_opt_selection(self):
        sampling = np.arange(1, self.test_range, 1)
        selective = [bernoulli_success_probability(self.n_min_transmitters, self.n_slots, s) for s in sampling]
        self._store_results_to_file(sampling, selective)
        self._plot_results(sampling, selective)
        print(f'sampling: {" & ".join(map(lambda i: str(i), sampling))}')
        print(f'results: {" & ".join(map(lambda v: format(v, ".4f"), selective))}')

    def _store_results_to_file(self, steps: np.ndarray, selective: List[float]) -> None:
        with open(os.path.join(self.experiment_dir, self.FILE_SINGLE_TABLE), 'w') as file:
            file.write(f'step\tselective\n')
            for stp in steps:
                file.write(f'{stp}\t{selective[stp-1]:.4f}\n')

    def _store_results_comparison_to_file(self, steps: np.ndarray, selective: List[float], simple: List[float]) -> None:
        with open(os.path.join(self.experiment_dir, self.FILE_TABLE), 'w') as file:
            file.write(f'step\tselective\tsimple\n')
            for stp in steps:
                file.write(f'{stp}\t{selective[stp]:.4f}\t{simple[stp]:.4f}\n')

    def _plot_results(self, sampling: np.ndarray, values: List[float]) -> None:
        plt.plot(sampling, values)
        plt.axis([0, self.test_range, 0, 1])
        plt.title(self.experiment_name)
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_SINGLE_FIGURE))
        plt.show()

    def _plot_results_comparison(self, sampling: np.ndarray, main: List[float], secondary: List[float]) -> None:
        plt.plot(sampling, main,
                 sampling, secondary, ':')
        plt.axis([0, self.n_slots, 0, 1])
        plt.title(self.experiment_name)
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_FIGURE))
        plt.show()

    @staticmethod
    def _selective_simplified_formula(n_slots: int, n_selections: int) -> float:
        return 1.0 - 1 / binomial(n_slots, n_selections)
