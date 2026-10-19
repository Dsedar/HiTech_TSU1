"""Rover implementation placeholder. Hardware commands belong in an adapter."""
from autonomy.contracts import RoverInput, RoverOutput


class Rover:
    name = "baseline"
    version = "0.1.0"

    def run(self, input: RoverInput) -> RoverOutput:
        raise NotImplementedError("Implement the RoverModule contract; see docs/module-contracts.md")
