"""Composition root. Hardware/simulator integration is intentionally not implemented."""

def main() -> None:
    """Load config, select module implementations, and run the scenario.

    TODO: Add a registry/orchestrator after the contracts and web-simulator adapter
    have been agreed by the team. This scaffold must not trigger hardware actions.
    """
    raise NotImplementedError(
        "Repository scaffold only: connect the web-simulator adapter and orchestrator first."
    )


if __name__ == "__main__":
    main()
