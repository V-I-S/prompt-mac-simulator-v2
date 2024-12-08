import os
import re
from typing import List, Pattern, Dict

from matplotlib import pyplot as plt
from numpy import arange

from tdma_sim_v2.lab.execution import Execution


class Comparison:
    FILE_FIGURE = 'comparison_visualization.png'
    BT_FILE = re.compile(r'result_per_topology_evolution_BtNode.*.tsv')
    TSLOTS_FILE = re.compile(r'result_per_topology_evolution_tSlotsNode.*.tsv')

    def __init__(self, execution: Execution, config: dict):
        self.execution = execution
        self.config = config
        self.n_slots = config['experiment.available-slots']
        self.experiment_size = config['experiment.size.stop']
        self.experiment_dir = config['experiment.dir']
        self.strategy_evolution_rng: range = range(config['experiment.strategy.evolution.start'],
                                                   config['experiment.strategy.evolution.stop'] + 1,
                                                   max(1, config['experiment.strategy.evolution.step']))

    def plot(self):
        self._plot_experiments(self._determine_file(self.BT_FILE), self._determine_file(self.TSLOTS_FILE))

    def _determine_file(self, regex: Pattern) -> str:
        selected_files = []
        for root, dirs, files in os.walk(self.experiment_dir):
            for file in files:
                if regex.match(file):
                    selected_files.append(file)
        if len(selected_files) != 1:
            print(f"Warning, not a single file was found for {regex}!")
        return selected_files[0]

    def _plot_experiments(self, bt_file: str, tslots_file: str) -> None:
        """Any model derivations in plot mechanism comes from the file naming, so the comparisons may be easily made."""
        print(f'Summarizing results in the plot...')
        ax = self._prepare_plot()
        print("Adding BT data to plot...")
        bt_data = self._extract_data("BtNode", bt_file)
        self._add_bt_to_plot(ax, bt_data)
        print("Adding tSlots data to plot...")
        tslots_data = self._extract_data("tSlotsNode", tslots_file)
        self._add_tslots_to_plot(ax, tslots_data)
        self._finalize_plot(ax)

    def _extract_data(self, model: str, file: str):
        complete_file = file.replace('per_topology_evolution', 'full')
        data = self.execution.get_experiment_results(file, model, complete_file)
        if model == "BtNode":
            data['index'] = [idx * self.n_slots for idx in data['index']]
        return data

    @staticmethod
    def _prepare_plot():
        fig, ax = plt.subplots()
        return ax

    @staticmethod
    def _add_bt_to_plot(ax, experiment: Dict[str, List[float]]):
        print(f'Plotting experiment for comparison...')
        print(experiment['index'])
        print(experiment['stderr'])
        return ax.errorbar(experiment['index'], experiment['value'], experiment['stderr'], fmt='o', markersize=2.0,
                            label="Bernoulli-Trials Strategy")

    @staticmethod
    def _add_tslots_to_plot(ax, experiment: Dict[str, List[float]]):
        print(f'Plotting experiment for comparison...')
        # print(experiment['index'])
        # print(experiment['stderr'])
        return ax.errorbar(experiment['index'], experiment['value'], experiment['stderr'], fmt='o', markersize=1.0,
                            label="t-Slots Strategy")

    def _finalize_plot(self, ax):
        plt.axis([0, 60, 0, 1])
        plt.title(f'BT and t-Slots Performance: Network of {self.experiment_size} Agents')
        ax.set_ylabel('Communication success probability')
        ax.set_xlabel('Expected number of transmissions per agent (BT: p•n, t-Slots: t)')

        ax.legend(loc='upper right')
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_FIGURE))
        plt.show()
