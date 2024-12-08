import logging
import os
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from numpy import sqrt

from tdma_sim_v2.config.execution import BERNOULLI_TRIES_RESOLUTION
from tdma_sim_v2.utils.dataSource import DataSource
from tdma_sim_v2.utils.exceptions.sanityCheckError import SanityCheckError


def _results_per_step(df: pd.DataFrame):
    return [df['#-comm-success'][df['step'] == step] for step in df['step'].unique()]


def _load_dataframe(datafile: str, skip_rows: int) -> pd.DataFrame:
    return pd.read_csv(datafile, delimiter='\t', skiprows=skip_rows, index_col=False)


class Comparison:
    FILE_FIGURE = 'comparison_visualization.png'
    NODE_BT = 'BtNode'
    NODE_TSLOTS = 'tSlotsNode'

    logger = logging.getLogger(__name__)

    def __init__(self, config: dict):
        self.config = config
        self.experiment_name = config['experiment.name']
        self.experiment_dir = config['experiment.dir']
        self.n_slots = config['experiment.available-slots']
        self.n_transmitters = config['experiment.size.start']
        self.n_max_transmitters = config.get('experiment.size.stop', None)
        self.n_accepted_failures = config['experiment.failures-tolerance.start']
        self.n_max_accepted_failures = config.get('experiment.failures-tolerance.stop', None)
        self.tick_last = config['experiment.output.ticks.last']
        self.tick_step = config['experiment.output.ticks.step']
        self.output_file = config['experiment.output.file']
        self.title = config['experiment.output.title']
        self.xaxis = config['experiment.output.axis.x']
        self.yaxis = config['experiment.output.axis.y']
        self.bt_resolution = config.get('experiment.bernoulli-tries-resolution', BERNOULLI_TRIES_RESOLUTION)
        self.data_source_1 = DataSource(config, 1) if 'experiment.data.1.path' in config else None
        self.data_source_2 = DataSource(config, 2) if 'experiment.data.2.path' in config else None
        self.data_source_3 = DataSource(config, 3) if 'experiment.data.3.path' in config else None
        self.data_source_4 = DataSource(config, 4) if 'experiment.data.4.path' in config else None

    def sanity_check(self):
        if self.n_max_transmitters is not None and self.n_transmitters != self.n_max_transmitters:
            raise SanityCheckError('Plot configuration operates exclusively on the starting number of transmitters')
        if self.n_accepted_failures is not None and self.n_max_accepted_failures != self.n_max_accepted_failures:
            raise SanityCheckError('Plot configuration operates exclusively on the starting failure tolerance limit')

    def compare(self):
        data1 = self.load_data(self.data_source_1)
        data2 = self.load_data(self.data_source_2)
        data3 = self.load_data(self.data_source_3)
        data4 = self.load_data(self.data_source_4)
        print(f'{self.data_source_1}\n{data1}')
        print(f'{self.data_source_2}\n{data2}')
        print(f'{self.data_source_3}\n{data3}')
        print(f'{self.data_source_4}\n{data4}')
        self.render_plot(data1, data2, data3, data4)

    def load_data(self, source: DataSource) -> Dict[str, List[float]]:
        if source is None:
            return {"index": [], "value": []}
        if source.is_calc():
            return self.interpolate_with_zeros(self.load_calculation(source.path))
        if source.is_exp():
            return self.load_experiment(source.path)
        raise ValueError(f"Unsupported data source type: {source.type}")

    def load_calculation(self, datafile: str) -> Dict[str, List[float]]:
        if datafile.find(self.NODE_BT) != -1:
            return self.load_bt_calculation_data(datafile)
        if datafile.find(self.NODE_TSLOTS) != -1:
            return self.load_tslots_calculation_data(datafile)
        raise ValueError(f"Strategy unrecognized for file: {datafile}")

    def load_bt_calculation_data(self, datafile: str) -> Dict[str, List[float]]:
        df = _load_dataframe(datafile, 1)
        self.logger.debug(f'BT: \n {df}')
        index = df['index'].apply(lambda idx: idx * self.n_slots).tolist()
        values = df['success_probability'].tolist()
        return {"index": index, "value": values}

    def load_tslots_calculation_data(self, datafile: str) -> Dict[str, List[float]]:
        df = _load_dataframe(datafile, 1)
        self.logger.debug(f'tSlots: \n {df}')
        index = df['index'].tolist()
        values = df['success_probability'].tolist()
        return {"index": index, "value": values}

    def load_experiment(self, datafile: str) -> Dict[str, List[float]]:
        if datafile.find(self.NODE_BT) != -1:
            return self.load_bt_experiment_data(datafile)
        if datafile.find(self.NODE_TSLOTS) != -1:
            return self.load_tslots_experiment_data(datafile)
        raise ValueError(f"Strategy unrecognized for file: {datafile}")

    def load_bt_experiment_data(self, datafile: str) -> Dict[str, List[float]]:
        df = _load_dataframe(datafile, 0)
        df = df[df['network-size'] == self.n_transmitters]
        df['step'] = df['step'].apply(lambda v: v / BERNOULLI_TRIES_RESOLUTION * self.n_slots)
        self.logger.debug(f'Loaded BT experiments data, #rows: \n {df.size}')
        return self._collect_experiment_results(df)

    def load_tslots_experiment_data(self, datafile: str) -> Dict[str, List[float]]:
        df = _load_dataframe(datafile, 0)
        df = df[df['network-size'] == self.n_transmitters]
        self.logger.debug(f'Loaded tSlots experiments data, #rows: \n {df.size}')
        return self._collect_experiment_results(df)

    def render_plot(self, data1, data2, data3, data4) -> None:
        self._inject_plot_data(data1, data2, data3, data4)
        self._prepare_plot_layout()
        plt.savefig(os.path.join(self.experiment_dir, self.output_file), dpi=550)
        plt.show()

    def _prepare_plot_layout(self):
        plt.title(self.title.replace('\\n', '\n'))
        plt.xlabel(self.xaxis)
        plt.ylabel(self.yaxis)
        plt.axis([0, self.tick_last, 0, 1])
        plt.xticks([r for r in range(0, self.tick_last + 1, self.tick_step)],
                   [r for r in range(0, self.tick_last + 1, self.tick_step)])
        plt.legend(loc='upper right')

    def _inject_plot_data(self, data1: Dict[str, List[float]], data2: Dict[str, List[float]],
                          data3: Dict[str, List[float]], data4: Dict[str, List[float]]):
        if self.data_source_1 is not None:
            frq = self.data_source_1.frequency
            plt.plot(data1['index'][0:-1:frq], data1['value'][0:-1:frq], color='#A6A7B5', linestyle='dashed',
                     label=self.data_source_1.label)
        if self.data_source_2 is not None:
            frq = self.data_source_2.frequency
            plt.plot(data2['index'][0:-1:frq], data2['value'][0:-1:frq], color='#1313FF',
                     label=self.data_source_2.label)
        if self.data_source_3 is not None:
            frq = self.data_source_3.frequency
            clr = 'black' if self.data_source_2 is not None else '#A6A7B5'
            plt.errorbar(data3['index'][0:-1:frq], data3['value'][0:-1:frq], yerr=data3['stderr'][0:-1:frq], fmt='o',
                         markersize=0.5, color=clr, label=self.data_source_3.label)
        if self.data_source_4 is not None:
            frq = self.data_source_4.frequency
            clr = 'black' if self.data_source_1 is not None else '#1313FF'
            plt.errorbar(data4['index'][0:-1:frq], data4['value'][0:-1:frq], yerr=data4['stderr'][0:-1:frq], fmt='o',
                         markersize=0.5, color=clr, label=self._label_4th_if_unique())

    def _label_4th_if_unique(self) -> Optional[str]:
        label = self.data_source_4.label
        other_sources = [self.data_source_1, self.data_source_2, self.data_source_3]
        other_labels = map(lambda source: source.label,
                           filter(lambda source: source is not None, other_sources))
        return label if label not in other_labels else None

    def interpolate_with_zeros(self, data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        max_arg = data["index"][-1]
        gap = int(self.tick_last - max_arg) + 2
        return {"index": data["index"] + [max_arg + i for i in range(0, gap)],
                "value": data["value"] + [0.0] * gap}

    @staticmethod
    def _collect_experiment_results(df: pd.DataFrame) -> Dict[str, List[float]]:
        step_results = _results_per_step(df)
        step_repeats = [len(results) for results in step_results]
        index = df['step'].unique().tolist()
        values = [results.sum() / len(results) for results in step_results]
        err = [Comparison._calculate_standard_error(*data) for data in zip(values, step_repeats)]
        return {"index": index, "value": values, "stderr": err}

    @staticmethod
    def _calculate_standard_error(prob: float, repeats: int) -> float:
        std_dev = sqrt(prob - prob ** 2)
        return 3.290526731 * sqrt(std_dev ** 2 / repeats)
