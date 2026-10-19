"""Target selection placeholder; tie-breaking policy must be agreed by the team."""
from autonomy.contracts import SelectionInput, SelectionOutput


class TargetSelector:
    name = "max_red_count"
    version = "0.1.0"

    def run(self, input: SelectionInput) -> SelectionOutput:
        raise NotImplementedError("Implement the TargetSelectorModule contract.")
