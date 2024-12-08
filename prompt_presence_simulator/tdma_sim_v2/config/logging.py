import logging
from logging import config
from datetime import datetime

from tdma_sim_v2.config.execution import ExecutionConfig


class LoggingConfig:
    config: dict = {
        'version': 1,
        'root': {
            'handlers': ['file'],
            'level': 'INFO'
        },
        'prompt-mac': {
            'handlers': ['file'],
            'level': 'INFO'
        },
        'handlers': {
            'file': {
                'formatter': 'std_out',
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'filename': datetime.now().strftime('tdma_sim_v1_%Y-%m-%d_%H-%M-%S.log')
            }
        },
        'formatters': {
            'std_out': {
                'format': '[%(levelname)s] %(threadName)s | %(module)s:%(lineno)d : %(message)s',
            }
        },
    }

    @classmethod
    def configure(cls):
        if ExecutionConfig.config['experiment.dir']:
            logs_destination = f"{ExecutionConfig.config['experiment.dir']}/{cls.config['handlers']['file']['filename']}"
            cls.config['handlers']['file']['filename'] = logs_destination
        logging.config.dictConfig(cls.config)
