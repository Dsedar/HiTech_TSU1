"""Striker implementation placeholder. No flight-control logic is included."""
from autonomy.contracts import StrikerInput, StrikerOutput


class Striker:
    name = "baseline"
    version = "0.1.0"

    def run(self, input: StrikerInput) -> StrikerOutput:
        raise NotImplementedError("Implement the StrikerModule contract; see docs/module-contracts.md")
