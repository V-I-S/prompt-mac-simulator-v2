import itertools
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from tdma_calc.api import bernoulli_optimal_probability_choice, tslots_optimal_selects_choice

from tdma_sim_v2.utils.buffered_report_writer import BufferedReportWriter


class Optimal:
    FILE_FIGURE = 'min_network.png'
    BT_COLUMNS = (0, 1)
    TSLOTS_COLUMNS = (0, 4)

    logger = logging.getLogger(__name__)
    reporter: BufferedReportWriter

    def __init__(self, config: dict):
        self.experiment_name = config['experiment.name']
        self.experiment_dir = config['experiment.dir']
        self.n_min_transmitters = config['experiment.size.start']
        self.n_max_transmitters = config['experiment.size.stop'] + 1
        self.transmitters_step = config['experiment.size.step']
        self.n_slots = config['experiment.available-slots']
        self.target_success_prob = config['experiment.target-percent'] / 100

    def optimal(self) -> None:
        self.reporter = BufferedReportWriter(self.experiment_dir, 'report_optimal_20.asv',
                                             ['agents', 'n_bt', 'config_bt', 'prob_bt', 'n_tslots', 'config_tslots', 'prob_tslots'], 1, 'a')
        for agents in range(self.n_min_transmitters, self.n_max_transmitters, self.transmitters_step):
            self._find_optimal(agents, self.n_slots)
        self.reporter.close()

    def smallest_n_to_achieve_target(self) -> None:
        self.reporter = BufferedReportWriter(self.experiment_dir, 'report_' + str(self.target_success_prob) + '.asv',
                                             ['agents', 'slots', 'config_bt', 'prob_bt', 'config_tslots', 'prob_tslots'], 1, 'a')
        for agents in range(self.n_min_transmitters, self.n_max_transmitters, self.transmitters_step):
            self._find_smallest_n_to_achieve_target(agents)
        self.reporter.close()

    def plot_smallest_n(self):
        bernoulli = self._retrieve_calculated_values(self.BT_COLUMNS)
        selective = self._retrieve_calculated_values(self.TSLOTS_COLUMNS)

        fig, ax = plt.subplots()
        ax.plot(bernoulli['index'], bernoulli['value'], '--', label='Bernoulli-Trials strategy')
        ax.plot(selective['index'], selective['value'], '-', label='t-Slots strategy')
        ax.set_xlabel('Number of agents (k)')
        ax.set_xticks(list(range(0, int(bernoulli['index'][-1]+1), 2)))
        ax.set_ylabel('Communication round size (n)')
        # ax.set_ylabel(f'Minimal network size (n) that enables\ncommunication success probability {self.target_success_prob:0.1f}')
        ax.legend(loc='upper left')
        plt.axis([bernoulli['index'][0], bernoulli['index'][-1], 0, bernoulli['value'][-1]])
        plt.title(f'Models comparison, minimal communication round size\nto enable communication success probability {self.target_success_prob:0.1f}')
        plt.savefig(os.path.join(self.experiment_dir, self.FILE_FIGURE))
        plt.show()



    def _find_optimal(self, agents: int, slots: int) -> None:
        bernoulli = bernoulli_optimal_probability_choice(agents, slots)
        tslots = tslots_optimal_selects_choice(agents, slots)
        line = f'{agents} & {slots} & {bernoulli[0]:.4f} & {bernoulli[1]:.4f} & {tslots[0]} & {tslots[1]:.4f}'
        print(f'Agents: {agents}, Slots: {slots}\nBernoulli: {bernoulli}\nt-Slots: {tslots}')
        print(line)
        self.reporter.write(line)

    def _find_smallest_n_to_achieve_target(self, agents: int) -> None:
        bernoulli = self._find_n_bernoulli(agents)
        tslots = self._find_n_tslots(agents)
        line = f'{agents} & {bernoulli[0]} & {bernoulli[1]:0.4f} & {bernoulli[2]:0.4f} & {tslots[0]} & {tslots[1]} & {tslots[2]:0.4f}'
        print(f'Agents: {agents}, bernoulli slots: {bernoulli[0]}, tSlots: {tslots[0]}')
        self.reporter.write(line)

    def _find_n_bernoulli(self, agents: int) -> (int, float, float):
        for slots in itertools.count(start=self.n_slots):
            optimal = bernoulli_optimal_probability_choice(agents, slots)
            print(f'   Agents: {agents}, Slots: {slots}: Bernoulli reaches max {optimal[1]}')
            if optimal[1] >= self.target_success_prob:
                return (slots, *optimal)

    def _find_n_tslots(self, agents: int) -> (int, int, float):
        for slots in itertools.count(start=self.n_slots):
            optimal = tslots_optimal_selects_choice(agents, slots)
            print(f'   Agents: {agents}, Slots: {slots}: tSlots reaches max {optimal[1]}')
            if optimal[1] >= self.target_success_prob:
                return (slots, *optimal)

    def _retrieve_calculated_values(self, strategy_columns: Tuple[int, int], scaler: float = 1.0) -> Dict[str, List[float]]:
        results_file = self._detrmine_output_file()
        df = pd.read_csv(results_file, delimiter='&', header=None, skiprows=1)
        agents_index = df[strategy_columns[0]].tolist()
        slots_values = df[strategy_columns[1]].tolist()
        return {"index": agents_index, "value": slots_values}

    def _detrmine_output_file(self):
        dir = self.experiment_dir
        file = list(filter(lambda file: file.startswith('report_') and file.endswith('.asv'), os.listdir(dir)))[0]
        return os.path.join(dir, file)
