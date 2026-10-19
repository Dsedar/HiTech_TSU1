"""Scout implementation placeholder. Replace this class with a candidate algorithm."""
from autonomy.contracts import ScoutInput, ScoutOutput, ModuleStatus


class Scout:
    name = "baseline"
    version = "0.1.0"

    def run(self, input: ScoutInput) -> ScoutOutput:
        raise NotImplementedError("Implement the ScoutModule contract; see docs/module-contracts.md")
