# Patryk Stopyra
# Department of Fundamentals of Computer Science
# Wroclaw University of Technology
# 2021
#
# Uncontrolled channel access simulator (v3.0)
import sys

from tdma_sim_v2.config.execution import ExecutionConfig
from tdma_sim_v2.config.logging import LoggingConfig
from tdma_sim_v2.lab import suboptimal
from tdma_sim_v2.lab.comparison import Comparison
from tdma_sim_v2.lab.custom import Custom
from tdma_sim_v2.lab.execution import Execution
from tdma_sim_v2.lab.experiment import Experiment
from tdma_sim_v2.lab.optimal import Optimal
from tdma_sim_v2.lab.suboptimal import Suboptimal
from tdma_sim_v2.lab.summary import Summary
from tdma_sim_v2.utils.exceptions.sanityCheckError import SanityCheckError

if __name__ == '__main__':
    print('Loading experiment configuration...')
    ExecutionConfig.configure('tests/' + sys.argv[1] + '/config.yml')

    print('Loading logging configuration...')
    LoggingConfig.configure()

    commands = sys.argv[2].split(',')
    if 'experiment' in commands:
        print('Loading module: execution...')
        exe = Execution(ExecutionConfig.config)
        print('Starting experiment execution...')
        try:
            exe.perform()
        except SanityCheckError as e:
            print(str(e) + '. Check configuration: ' + e.reason)
        except BaseException as e:
            raise e
    if 'summarize' in commands:
        print('Loading module: summary...')
        summary = Summary(ExecutionConfig.config)
        print('Starting summary calculation...')
        try:
            summary.plot()
        except ValueError as e:
            raise e
    if 'compare2' in commands:
        print('Loading module: custom...')
        custom = Custom(ExecutionConfig.config)
        print('Starting custom 2-transmitter comparison...')
        try:
            custom.compare_2_transmitters()
        except ValueError as e:
            raise e
    if 'custom' in commands:
        print('Loading module: custom...')
        custom = Custom(ExecutionConfig.config)
        print('Starting custom selects calculation to find optimal configuration...')
        try:
            custom.test_transmitters_for_opt_selection()
        except ValueError as e:
            raise e
    if 'optimal' in commands:
        print('Loading module: optimal...')
        optimal = Optimal(ExecutionConfig.config)
        print('Starting custom selects calculation to find optimal configuration...')
        try:
            optimal.optimal()
        except ValueError as e:
            raise e
    if 'meet' in commands:
        print('Loading module: optimal...')
        optimal = Optimal(ExecutionConfig.config)
        print('Starting custom calculation to find the minimal n to meet the target...')
        try:
            optimal.smallest_n_to_achieve_target()
        except ValueError as e:
            raise e
    if 'meet_plot' in commands:
        print('Loading module: optimal...')
        optimal = Optimal(ExecutionConfig.config)
        print('Starting of plotting the minimal n to meet target...')
        try:
            optimal.plot_smallest_n()
        except ValueError as e:
            raise e
    if 'suboptimal' in commands:
        print('Loading module: suboptimal...')
        suboptimal = Suboptimal(ExecutionConfig.config)
        print('Starting custom calculation of the performance in the suboptimal setup...')
        try:
            suboptimal.suboptimal()
        except ValueError as e:
            raise e
    if 'suboptimal_plot' in commands:
        print('Loading module: suboptimal...')
        suboptimal = Suboptimal(ExecutionConfig.config)
        print('Starting of plotting the suboptimal setup resilience...')
        try:
            suboptimal.suboptimal_plot()
        except ValueError as e:
            raise e
    if 'compare' in commands:
        print('Loading module: comparison...')
        comp = Comparison(ExecutionConfig.config)
        print('Comparing 2 provided data files...')
        try:
            comp.compare()
        except ValueError as e:
            raise e
    print(f'Execution finished.')
