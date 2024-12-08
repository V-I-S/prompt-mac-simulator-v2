import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from prompt_mac_calculator.api import fault_tolerant_bernoulli_success_probability, \
    fault_tolerant_tslots_success_probability, bernoulli_optimal_probability_choice, tslots_optimal_selects_choice

from tdma_sim_v2.utils.buffered_report_writer import BufferedReportWriter


class Suboptimal:
    FILE_FIGURE = 'suboptimal_total_{}_slots_{}_agents{}.png'
    SUBOPTIMAL_FILE = 'report_suboptimal.asv'
    OPTIMAL_FILE = 'report_optimal.asv'
    BT_COLUMNS = (0, 3)
    TSLOTS_COLUMNS = (0, 5)

    logger = logging.getLogger(__name__)
    reporter_sub: BufferedReportWriter
    reporter_opt: BufferedReportWriter

    def __init__(self, config: dict):
        self.experiment_name = config['experiment.name']
        self.experiment_dir = config['experiment.dir']
        self.n_min_transmitters = config['experiment.size.start']
        self.n_max_transmitters = config['experiment.size.stop'] + 1
        self.transmitters_step = config['experiment.size.step']
        self.n_slots = config['experiment.available-slots']
        self.tolerance = config['experiment.failures-tolerance.start']
        self.agents_optimal_calc = config['experiment.agents-for-optimal-calc']

    def suboptimal(self):
        self.reporter_sub = BufferedReportWriter(self.experiment_dir, Suboptimal.SUBOPTIMAL_FILE,
                                             ['agents', 'config_bt', 'prob_bt', 'config_tslots', 'prob_tslots'], 1, 'a')
        self.reporter_opt = BufferedReportWriter(self.experiment_dir, Suboptimal.OPTIMAL_FILE,
                                             ['agents', 'config_bt', 'prob_bt', 'config_tslots', 'prob_tslots'], 1, 'a')
        bt = bernoulli_optimal_probability_choice(self.agents_optimal_calc, self.n_slots, self.tolerance)[0]
        tslots = tslots_optimal_selects_choice(self.agents_optimal_calc, self.n_slots, self.tolerance)[0]
        print(f'Optimal p = {bt}, optimal t = {tslots}, fault tolerance = {self.tolerance}')
        for agents in range(self.n_min_transmitters, self.n_max_transmitters, self.transmitters_step):
            self._calculate_suboptimal_results(agents, self.n_slots, bt, tslots, self.tolerance)
            self._calculate_optimal_results(agents, self.n_slots, self.tolerance)
        self.reporter_sub.close()
        self.reporter_opt.close()

    def suboptimal_plot(self):
        bernoulli_sub = self._retrieve_calculated_values(Suboptimal.SUBOPTIMAL_FILE, self.BT_COLUMNS)
        tslots_sub = self._retrieve_calculated_values(Suboptimal.SUBOPTIMAL_FILE, self.TSLOTS_COLUMNS)
        bernoulli_opt = self._retrieve_calculated_values(Suboptimal.OPTIMAL_FILE, self.BT_COLUMNS)
        tslots_opt = self._retrieve_calculated_values(Suboptimal.OPTIMAL_FILE, self.TSLOTS_COLUMNS)
        faults_label = f", {self.tolerance} accepted fault" if self.tolerance > 0 else ""

        fig, ax = plt.subplots()
        ax.plot(bernoulli_sub['index'], bernoulli_sub['value'], '--', color='#A6A7B5', label='BT strategy')
        ax.plot(tslots_sub['index'], tslots_sub['value'], '-', color='#1313FF', label='t-Slots strategy')
        ax.plot(bernoulli_opt['index'], bernoulli_opt['value'], '.', color='#A6A7B5',
                label='BT strategy (config. optimally)')
        ax.plot(tslots_opt['index'], tslots_opt['value'], '.', color='#1313FF',
                label='t-Slots strategy (config. optimally)')
        ax.set_xlabel('Actual number of agents (k)')
        ax.set_xticks(list(range(0, int(bernoulli_sub['index'][-1]+1), 2)))
        ax.set_ylabel('Synchronization probability')
        ax.legend(loc='upper right')
        plt.axis([bernoulli_sub['index'][0], bernoulli_sub['index'][-1]//2, 0, 1])

        plt.title(f'Strategies comparison, the suboptimal configuration\n'
                  f'(configured as the optimal for {self.agents_optimal_calc} slots{faults_label})')
        plt.savefig(self._generate_plot_filename(), dpi=550)
        plt.show()

    def _calculate_suboptimal_results(self, agents: int, slots: int, bt_config: float, tslots_config: int,
                                      tolerance: int):
        bt_prob = fault_tolerant_bernoulli_success_probability(agents, slots, bt_config, tolerance)
        tslots_prob = fault_tolerant_tslots_success_probability(agents, slots, tslots_config, tolerance)
        print(f'Agents: {agents}, Slots: {slots} suboptimal:\nBernoulli({bt_config}): {bt_prob}\nt-Slots({tslots_config}): {tslots_prob}')
        self.reporter_sub.write(f'{agents} & {slots} & {bt_config:.4f} & {bt_prob:.4f} & {tslots_config} & {tslots_prob:.4f}')

    def _calculate_optimal_results(self, agents: int, slots: int, tolerance: int):
        bt = bernoulli_optimal_probability_choice(agents, slots, tolerance)
        try:
            tslots = tslots_optimal_selects_choice(agents, slots, tolerance)
        except ValueError as ex:
            self.logger.warning("Sanity check failed for t-Slots", ex)
            tslots = 0, 1.0
        print(f'Agents: {agents}, Slots: {slots} optimal:\nBernoulli({bt[0]}): {bt[1]}\nt-Slots({tslots[0]}): {tslots[1]}')
        self.reporter_opt.write(f'{agents} & {slots} & {bt[0]:.4f} & {bt[1]:.4f} & {tslots[0]} & {tslots[1]:.4f}')

    def _retrieve_calculated_values(self, filename: str, strategy_columns: Tuple[int, int]) -> Dict[str, List[float]]:
        results_file = os.path.join(self.experiment_dir, filename)
        df = pd.read_csv(results_file, delimiter='&', header=None, skiprows=1)
        agents_index = df[strategy_columns[0]].tolist()
        slots_values = df[strategy_columns[1]].tolist()
        return {"index": agents_index, "value": slots_values}

    def _generate_plot_filename(self):
        fault_tolerance = f"_{self.tolerance}_faults" if self.tolerance > 0 else ""
        return os.path.join(self.experiment_dir,
                            self.FILE_FIGURE.format(self.n_slots, self.agents_optimal_calc, fault_tolerance))
