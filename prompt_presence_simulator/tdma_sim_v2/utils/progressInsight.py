from datetime import datetime


class ProgressInsight:
    def __init__(self, destinated_logger):
        self.logger = destinated_logger
        self.milestone = 0
        self.progress_bar = 0

    def intro(self, banner: str, description: str) -> None:
        print(f'Experiments suite {banner}\n{description}')
        print(f'Start time: {datetime.now()}\nProgress:')
        self.logger.info("Started experiments suite for: %s", banner)

    def finalizing(self, banner: str, tests_n: int) -> None:
        print(f'Finished all experiments. Number of executed tests: {tests_n}')
        print('Finalizing reports...')
        self.logger.info("Finalizing experiments' reports for: %s", banner)

    def finished(self, banner: str) -> None:
        print('Experiments suite has finished')
        print(f'End time: {datetime.now()}')
        self.logger.info("Finished experiments suite for: %s", banner)

    def phase_begin(self, size: int, topology_step: int, iteration: int) -> None:
        self.milestone = iteration
        print(f'Size:{size}, Top-evo:{topology_step}:')
        print(f'{0:6d} ', end='', flush=True)

    def phase_end(self):
        self.progress_bar = 0
        print()

    def progress(self, iteration: int) -> None:
        print('\b' * (self.progress_bar + 7), end='')
        step = iteration - self.milestone
        if step % 1000 == 1:
            self.progress_bar += 1
        print(f'{step:6d} ' + '|' * self.progress_bar, end='', flush=True)
