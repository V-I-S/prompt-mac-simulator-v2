from typing import List

import tdma_sim_v2.model.nodes.node as n


class MessageBody:
    def __init__(self):
        pass

    def __str__(self) -> str:
        return 'Base MessageBody'


class Message:
    def __init__(self, body: MessageBody, source: 'n.Node'):
        self.body = body
        self.source = source

    def __str__(self) -> str:
        return f'Message from N:{self.source.node_id}'


class UnicastMessage(Message):
    def __init__(self, body: MessageBody, source: 'n.Node', target: 'n.Node'):
        super().__init__(body, source)
        self.target = target

    def __str__(self) -> str:
        return f'UnicastMessage from N:{self.source.node_id} to N:{self.target.node_id}'


class MulticastMessage(Message):
    def __init__(self, body: MessageBody, source: 'n.Node', target: List['n.Node']):
        super().__init__(body, source)
        self.target = target

    def __str__(self) -> str:
        targets = ','.join([str(node.node_id) for node in self.target])
        return f'MulticastMessage from N:{self.source.node_id} to N:{targets}'
