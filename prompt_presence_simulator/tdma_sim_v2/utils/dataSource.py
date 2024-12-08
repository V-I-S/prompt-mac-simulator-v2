import os


class DataSource:
    type: str
    label: str
    path: str
    frequency: int

    def __init__(self, config: dict, idx: int):
        self.type = config['experiment.data.' + str(idx) + '.type']
        self.label = config['experiment.data.' + str(idx) + '.label']
        self.frequency = self._read_frequency(config, idx)
        self.path = self._read_path(config, idx)

    def __str__(self):
        return f'{self.type}: {self.label}, {self.path}'

    def is_calc(self) -> bool:
        return self.type == 'CALCULATION'

    def is_exp(self) -> bool:
        return self.type == 'EXPERIMENT'

    @staticmethod
    def _read_frequency(config: dict, idx: int) -> int:
        key = 'experiment.data.' + str(idx) + '.frequency'
        if key not in config:
            return 1
        return int(config[key])

    @staticmethod
    def _read_path(config: dict, idx: int) -> str:
        declared_path = config['experiment.data.' + str(idx) + '.path']
        if os.path.basename(declared_path) == declared_path:
            return os.path.join(config['experiment.dir'], declared_path)
        return declared_path