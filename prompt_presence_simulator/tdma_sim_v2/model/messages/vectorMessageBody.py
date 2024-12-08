from typing import List

from tdma_sim_v2.model.messages.message import MessageBody


class VectorMessageBody(MessageBody):
    def __init__(self, slots: List[int]):
        super().__init__()
        self.vector = slots

    def __str__(self) -> str:
        return f"slots={self.vector}"
