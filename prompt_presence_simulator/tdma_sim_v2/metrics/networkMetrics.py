import logging
from typing import List, Tuple


class NetworkMetrics:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self._n_messages = 0
        self._n_deliveries = 0
        self._sentiment_dynamics = []

    def count_message(self) -> None:
        self._n_messages += 1

    def get_messages_number(self) -> int:
        return self._n_messages

    def count_delivery(self, new_deliveries: int = 1) -> None:
        self._n_deliveries += new_deliveries

    def get_deliveries_number(self) -> int:
        return self._n_deliveries

    def record_sentiment(self, sentiment: float) -> None:
        entry = (self._n_messages, self._n_deliveries, sentiment)
        if len(self._sentiment_dynamics) > 0 and entry == self._sentiment_dynamics[-1]:
            self.logger.debug('Sentiment recorded already before: messages=%s, deliveries=%s, sentiment=%s', *entry)
            return
        self._sentiment_dynamics += [entry]

    def get_sentiment_dynamics(self) -> List[Tuple[int, int, float]]:
        """@:returns List of tuples (messages, deliveries, sentiment) representing the historical change of sentiment"""
        return self._sentiment_dynamics
