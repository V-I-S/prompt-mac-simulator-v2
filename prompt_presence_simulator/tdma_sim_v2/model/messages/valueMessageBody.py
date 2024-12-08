from tdma_sim_v2.model.messages.message import MessageBody


class ValueMessageBody(MessageBody):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return f"value={self.value}"
