import logging
from typing import List, Optional

from tdma_sim_v2.model.networks.network import Network


class DistributedConsensusEvaluator:
    logger = logging.getLogger(__name__)

    def __init__(self):
        pass

    def evaluate(self, network: Network, valuation: List[float]) -> Optional[str]:
        verdict = []
        if not self.check_communication_success(network):
            verdict += ['comm-unsuccessful']
        if len(verdict):
            return '|'.join(verdict)
        else:
            return None

    def check_communication_success(self, network: Network, failures_tolerance: int = 0) -> bool:
        """Chcks if all nodes are certain about the shared opinion"""
        communication_successful = network.is_communication_success(failures_tolerance)
        self.logger.debug('Communication success: %s', communication_successful)
        return communication_successful

    def check_communication_success_percentile(self, network: Network, percentile: float) -> bool:
        """Chcks if all nodes are certain about the shared opinion"""
        communication_successful = network.is_communication_success_percentile(percentile)
        self.logger.debug('Communication success (percentile %f): %s', communication_successful)
        return communication_successful

    def check_common_tilt(self, network: Network) -> bool:
        """Check if all nodes are skewed toward the same opinion (converged in Benezit sense)"""
        has_common_tilt = network.has_common_tilt()
        self.logger.debug('Network has common tilt: %s', has_common_tilt)
        return has_common_tilt

    def check_tilt_correct(self, network: Network, valuation: List[float]) -> bool:
        """All nodes are converged in the sense of Benezit and the common opinion is tilted towards the initial majority"""
        if not network.has_common_tilt():
            return False
        network_sentiment = network.get_sentiment()
        initial_sentiment = float(sum(valuation))
        self.logger.debug('Network converged to the initial majority: %b', _has_same_sign(network_sentiment, initial_sentiment))
        return _has_same_sign(network_sentiment, initial_sentiment)


def _has_same_sign(a: float, b: float) -> bool:
    """Checks if two numbers has the same sign - allows to define if initial sentiment had the same tilt as the result one from the network"""
    return a * b > 0
