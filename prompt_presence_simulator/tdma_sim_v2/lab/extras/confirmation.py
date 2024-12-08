import os
import re
from typing import List, Pattern, Dict

from matplotlib import pyplot as plt
from numpy import arange

from tdma_sim_v2.lab.execution import Execution


class Confirmation:
    FILE_FIGURE = 'comparison_visualization.png'
    EXPERIMENT_FILE_REGEX = re.compile(r'result_per_topology_evolution_.*.tsv')
    CALCULATION_FILE_REGEX = re.compile(r'result_calculation_data_.*.tsv')

    def __init__(self, execution: Execution, config: dict):
        self.execution = execution
        self.config = config
        self.experiment_dir = config['experiment.dir']
        self.experiment_model = self.config['experiment.model.node']
        self.strategy_evolution_rng: range = range(config['experiment.strategy.evolution.start'],
                                                   config['experiment.strategy.evolution.stop'] + 1,
                                                   max(1, config['experiment.strategy.evolution.step']))
    def plot(self):
        self._plot_experiments(self._determine_files(self.EXPERIMENT_FILE_REGEX),
                               self._determine_files(self.CALCULATION_FILE_REGEX))


    def _determine_files(self, regex: Pattern) -> List[str]:
        selected_files = []
        for root, dirs, files in os.walk(self.experiment_dir):
            for file in files:
                if regex.match(file):
                    selected_files.append(file)
        return selected_files

    def _plot_experiments(self, experiments: List[str], calculations: List[str]) -> None:
        """Any model derivations in plot mechanism comes from the file naming, so the comparisons may be easily made."""
        print(f'Summarizing results in the plot...')
        ax = self._prepare_plot()
        lines = []
        if self.experiment_model == 'BtNode':
            calculations.reverse()  # reverse, as BT for some reson mixes the order
        for file in calculations:
            # tolerance = int(self.calculation_file_regex.match(file).group(2))
            data = self.execution.get_calculation_results(file)
            lines.append(self._add_calculation_data_to_plot(ax, data))
        for file in experiments:
            model = self.experiment_model
            # tolerance = int(self.experiment_file_regex.match(file).group(2))
            complete_file = file.replace('per_topology_evolution', 'full')
            print(complete_file)
            data = self.execution.get_experiment_results(file, model, complete_file)
            lines.append(self._add_experiment_data_to_plot(ax, data))
        self._finalize_plot(ax, lines)

    def _prepare_plot(self):
        fig, ax = plt.subplots()
        ax.set_ylabel('Synchronization probability')
        return ax

    def _add_experiment_data_to_plot(self, ax, experiment: Dict[str, List[float]]):
        print(f'Plotting experiment for comparison...')

        self._interpolate_impossible_cases(experiment)
        print(experiment['index'])
        print(experiment['stderr'])

        return ax.errorbar(experiment['index'], experiment['value'], experiment['stderr'], fmt='o', markersize=1.0,
                           color='black', label=None)

    def _interpolate_impossible_cases(self, experiment: Dict[str, List[float]]):
        if experiment['stderr'][-1] == 0.0:
            experiment['stderr'] += [0.0] * (len(experiment['index']) - len(experiment['stderr']))

    def _add_calculation_data_to_plot(self, ax, calculation: Dict[str, List[float]]):
        print(f'Plotting calculation for comparison...')
        return ax.plot(calculation['index'], calculation['value'], label=None)

    def _finalize_plot(self, ax, lines: List):
        if self.experiment_model == 'BtNode':
            # plt.axis([0, 1, 0, 1])
            plt.axis([0, 1, 0, 1])  # extra
            # plt.xticks([r for r in arange(0.0, 0.41, 0.05)],
            #            ["0.00"] + [f'{r:0.2f}' for r in arange(0.05, 0.41, 0.05)])  # extra
            plt.title('Bernoulli-Trials Strategy')
            ax.set_xlabel('Agent\'s transmission probability in the single slot (p)')
        elif self.experiment_model == 'tSlotsNode':
            plt.axis([0, self.strategy_evolution_rng.stop, 0, 1])
            plt.xticks([r for r in range(0, 31, 5)], [r for r in range(0, 31, 5)])  # extra
            plt.title('t-Slots Strategy')
            ax.set_xlabel('Cardinality of agent\'s transmission set (t)')
        else:
            print('Error: Plotting issue, unrecognized agents'' strategy')
        # self._mathplot_label_test_121_tslots(lines)
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_FIGURE), dpi=550)
        plt.show()

    def _mathplot_label_test_123_bt(self, lines: List):
        lines[3][0].set_label("Fault-tolerance: 0")
        lines[3][0].set_color('#01199B')
        lines[2][0].set_label("Fault-tolerance: 1")
        lines[2][0].set_color('#1313ff')
        lines[0][0].set_label("Fault-tolerance: 2")
        lines[0][0].set_color('#5353bf')
        # lines[0][0].set_linestyle('--')
        lines[1][0].set_label("Fault-tolerance: 3")
        lines[1][0].set_color('#83838f')
        lines[4][0].set_label("Fault-tolerance: 4")
        lines[4][0].set_color('#C6C7D5')
        #lines[5][0].set_label("Fault-tolerance: 3")
        lines[6][0].set_label("Simulation data")
        plt.legend(loc='upper right')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [3,2,0,1,4,5]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    def _mathplot_label_test_121_tslots(self, lines: List):
        lines[2][0].set_label("Fault-tolerance: 0")
        lines[2][0].set_color('#01199B')
        lines[3][0].set_label("Fault-tolerance: 1")
        lines[3][0].set_color('#1313ff')
        lines[1][0].set_label("Fault-tolerance: 2")
        lines[1][0].set_color('#5353bf')
        # lines[0][0].set_linestyle('--')
        lines[0][0].set_label("Fault-tolerance: 3")
        lines[0][0].set_color('#83838f')
        lines[4][0].set_label("Fault-tolerance: 4")
        lines[4][0].set_color('#C6C7D5')
        #lines[5][0].set_label("Fault-tolerance: 3")
        lines[6][0].set_label("Simulation data")
        plt.legend(loc='upper right')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2,3,1,0,4,5]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    def _mathplot_label_test_138_tslots(self, lines: List):
        lines[0][0].set_label("No. agents: 4")
        lines[0][0].set_color('#01199B')
        lines[1][0].set_label("No. agents: 6")
        lines[1][0].set_color('#1313ff')
        lines[3][0].set_label("No. agents: 8")
        lines[3][0].set_color('#507CC9')
        # lines[0][0].set_linestyle('--')
        lines[2][0].set_label("No. agents: 10")
        lines[2][0].set_color('#83838f')
        plt.legend(loc='upper right')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,1,3,2]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    def _mathplot_label_test_139_bt(self, lines: List):
        lines[0][0].set_label("No. agents: 4")
        lines[0][0].set_color('#01199B')
        lines[1][0].set_label("No. agents: 6")
        lines[1][0].set_color('#1313ff')
        lines[3][0].set_label("No. agents: 8")
        lines[3][0].set_color('#507CC9')
        # lines[0][0].set_linestyle('--')
        lines[2][0].set_label("No. agents: 10")
        lines[2][0].set_color('#83838f')
        # lines[4][0].set_label("Fault-tolerance: 4")
        # lines[4][0].set_color('#C6C7D5')
        #lines[5][0].set_label("Fault-tolerance: 3")
        lines[4][0].set_label("Simulation data")
        plt.legend(loc='upper right')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,1,3,2,4]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    def _mathplot_label_test_140_tslots(self, lines: List):
        lines[2][0].set_label("No. agents: 4")
        lines[2][0].set_color('#01199B')
        lines[1][0].set_label("No. agents: 6")
        lines[1][0].set_color('#1313ff')
        lines[0][0].set_label("No. agents: 8")
        lines[0][0].set_color('#507CC9')
        # lines[0][0].set_linestyle('--')
        lines[3][0].set_label("No. agents: 10")
        lines[3][0].set_color('#83838f')
        # lines[4][0].set_label("Fault-tolerance: 4")
        # lines[4][0].set_color('#C6C7D5')
        #lines[5][0].set_label("Fault-tolerance: 3")
        lines[4][0].set_label("Simulation data")
        plt.legend(loc='upper right')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2,1,0,3,4]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
