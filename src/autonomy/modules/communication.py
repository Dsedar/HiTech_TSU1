"""Transport-neutral communication interface placeholder."""
from autonomy.contracts import Message, MessageResult


class Communication:
    name = "local_stub"
    version = "0.1.0"

    def run(self, input: Message) -> MessageResult:
        raise NotImplementedError("Implement transport in an adapter and preserve this contract.")
