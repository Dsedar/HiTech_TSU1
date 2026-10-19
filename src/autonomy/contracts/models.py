"""Shared data contracts. Keep algorithm-specific internals out of this file."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModuleStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    NO_VALID_TARGET = "no_valid_target"
    NOT_READY = "not_ready"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class Point2D:
    """Position in the team-defined local frame, in metres."""
    x_m: float
    y_m: float
    frame_id: str = "team_local"


@dataclass(frozen=True)
class MarkerCounts:
    red: int = 0
    green: int = 0
    yellow: int = 0

    def __post_init__(self) -> None:
        if min(self.red, self.green, self.yellow) < 0:
            raise ValueError("Marker counts cannot be negative")


@dataclass(frozen=True)
class ScoutInput:
    """One observation. `image_ref` may be a URI, frame key, or adapter-defined handle."""
    image_ref: str
    timestamp_s: float
    takeoff_origin: Point2D | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedSite:
    site_id: str
    position: Point2D | None
    counts: MarkerCounts
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class ScoutOutput:
    status: ModuleStatus
    sites: tuple[DetectedSite, ...] = ()
    diagnostic: str = ""


@dataclass(frozen=True)
class SelectionInput:
    sites: tuple[DetectedSite, ...]


@dataclass(frozen=True)
class SelectionOutput:
    status: ModuleStatus
    selected_site: DetectedSite | None = None
    diagnostic: str = ""


@dataclass(frozen=True)
class RoverInput:
    target_site: DetectedSite
    current_position: Point2D | None = None
    payload_ready: bool = True
    timeout_s: float = 120.0


@dataclass(frozen=True)
class RoverOutput:
    status: ModuleStatus
    current_position: Point2D | None = None
    payload_ready: bool = False
    diagnostic: str = ""


@dataclass(frozen=True)
class StrikerInput:
    target_site: DetectedSite
    launch_authorized: bool
    battery_remaining_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrikerOutput:
    status: ModuleStatus
    processed_red_markers: int = 0
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if self.processed_red_markers < 0:
            raise ValueError("processed_red_markers cannot be negative")


@dataclass(frozen=True)
class Message:
    message_id: str
    message_type: str
    schema_version: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class MessageResult:
    status: ModuleStatus
    message_id: str
    acknowledged: bool = False
    diagnostic: str = ""
