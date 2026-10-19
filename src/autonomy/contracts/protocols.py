"""Public module interfaces. All candidate implementations must satisfy these protocols."""
from typing import Protocol
from .models import (
    ScoutInput, ScoutOutput, SelectionInput, SelectionOutput,
    RoverInput, RoverOutput, StrikerInput, StrikerOutput,
    Message, MessageResult,
)


class ScoutModule(Protocol):
    def run(self, input: ScoutInput) -> ScoutOutput: ...


class TargetSelectorModule(Protocol):
    def run(self, input: SelectionInput) -> SelectionOutput: ...


class RoverModule(Protocol):
    def run(self, input: RoverInput) -> RoverOutput: ...


class StrikerModule(Protocol):
    def run(self, input: StrikerInput) -> StrikerOutput: ...


class CommunicationModule(Protocol):
    def run(self, input: Message) -> MessageResult: ...
