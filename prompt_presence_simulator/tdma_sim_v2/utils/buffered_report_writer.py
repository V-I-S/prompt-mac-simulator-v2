import logging
import os
from typing import List, Generic, TypeVar, Optional, TextIO, Any

from tdma_sim_v2.utils.exceptions.setupError import SetupError
from tdma_sim_v2.utils.exceptions.streamError import StreamError

T = TypeVar('T')


class BufferedReportWriter(Generic[T]):
    OPEN_MODE_HEADERS = 'w'
    OPEN_MODE_DATA = 'a'
    DATA_SEPARATOR = '\t'
    LINES_SEPARATOR = '\n'
    DEFAULT_BUFFER = 64

    logger = logging.getLogger(__name__)

    def __init__(self, location: str, file: str, columns: List[str],
                 buffer_size: int = DEFAULT_BUFFER, open_mode: str = OPEN_MODE_HEADERS):
        self.stream = self._initiate_file(location, file, columns)
        self.file = file
        self.buffer = []
        self.buffer_size = buffer_size

    def write(self, record: Any) -> None:
        self.buffer.append(str(record))
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def close(self) -> None:
        if self.buffer:
            self._flush()
        self.stream.close()

    def _flush(self) -> None:
        try:
            self.stream.write(self.LINES_SEPARATOR.join(self.buffer) + self.LINES_SEPARATOR)
        except OSError as err:
            logging.error('Could not flush the buffer to %s file. %s', self.file, err)
            raise StreamError(f'Error writing to {self.file} filling stream. {err}')
        self.buffer.clear()

    def _initiate_file(self, location: str, file: str, columns: List[str]) -> Optional[TextIO]:
        stream = None
        try:
            if not os.path.exists(location):
                os.makedirs(location, exist_ok=True)
            stream = open(os.path.join(location, file), self.OPEN_MODE_HEADERS)
            stream.write(self.DATA_SEPARATOR.join(columns) + self.LINES_SEPARATOR)
        except OSError as err:
            logging.error('Could not create %s/%s file. %s', os.path.abspath(location), file, err)
            raise SetupError(f'Failed to initiate buffered output for {file} file')
        logging.info('Successfully initiated %s/%s report file.', os.path.abspath(location), file)
        return stream
