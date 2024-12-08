from typing import Any

from tdma_sim_v2.utils.yamlUtils import read_yaml, flatten_yaml

BERNOULLI_TRIES_RESOLUTION = 10000

class ExecutionConfig:
    """
    Main execution configuration. Expected unique yaml's content is:
        experiment:
            name:
            dir:
            header:
            description:
            model:
                network:
                node:
                executor:
            size:
                start:
                stop:
                step:
            topology:
                generator:
                evolution:
                    start:
                    stop:
                    step:
                instances-per-evolution:
            valuation:
                strategy:
                generator:
                max-instances-per-topology:
                max-steps-per-instance:
            failures-tolerance:
                start:
                stop:
                step:
                mode: LIMIT | EXACT
            sampling-frequency:
            execution-cores:
            target-percent:
            agents-for-optimal-calc:
              available-slots:
            data:
              1:
                type:
                path:
                label:

            output:
              file:
              title:
              ticks:
                last:
                step:
              axis:
                x:
                y:
    """

    failures_tolerance_mode: str = 'LIMIT'

    config: dict = None

    @classmethod
    def configure(cls, file: str) -> None:
        cls.config = flatten_yaml(read_yaml(file))
        cls.failures_tolerance_mode = cls.config['experiment.failures-tolerance.mode']