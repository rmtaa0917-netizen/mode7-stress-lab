"""joblab.models

Execution is at-least-once and external effects require idempotency.

Core data models, typed dataclasses, string enums, and validation invariants
for distributed job scheduling and coordination in JobLab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# =====================================================================
# Error Hierarchy
# =====================================================================

class JobLabError(Exception):
    """Base exception for all JobLab failures."""


class ValidationError(JobLabError):
    """Raised when an entity, parameter, or payload violates invariant constraints."""


class ConflictError(JobLabError):
    """Raised when a state transition or concurrency condition conflicts."""


class DependencyError(JobLabError):
    """Raised when job dependencies contain cycles, duplicates, self-references, or unresolvable states."""


class LeaseError(JobLabError):
    """Raised when lease acquisition, renewal, or release fails or is invalid."""


class CorruptionError(JobLabError):
    """Raised when persisted logs, state representations, or payloads are corrupt."""


# =====================================================================
# String Enums
# =====================================================================

class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    EXPIRED = "EXPIRED"


class EventType(str, Enum):
    SUBMITTED = "SUBMITTED"
    CLAIMED = "CLAIMED"
    LEASE_RENEWED = "LEASE_RENEWED"
    STARTED = "STARTED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    LEASE_RECOVERED = "LEASE_RECOVERED"


class FailureKind(str, Enum):
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"
    DEADLINE = "DEADLINE"
    CANCELLED = "CANCELLED"
    WORKER_CRASH = "WORKER_CRASH"


# =====================================================================
# Invariant Validation Routines
# =====================================================================

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-\.:]+$")


def validate_safe_id(value: Any, name: str = "id") -> str:
    """Ensure value is a non-empty, safe string identifier matching [A-Za-z0-9_\\-\\.:]+."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value).__name__}")
    if not value or not _SAFE_ID_RE.fullmatch(value):
        raise ValidationError(
            f"{name} must be a non-empty string consisting only of alphanumeric chars, hyphens, underscores, dots, or colons: {value!r}"
        )
    return value


def validate_optional_safe_id(value: Any, name: str = "id") -> Optional[str]:
    """Ensure value is None or a safe identifier."""
    if value is None:
        return None
    return validate_safe_id(value, name)


def validate_finite_time(value: Any, name: str = "time") -> float:
    """Ensure value is a finite, non-boolean real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a real number, got {type(value).__name__}")
    f_val = float(value)
    if not math.isfinite(f_val):
        raise ValidationError(f"{name} must be finite: {value!r}")
    return f_val


def validate_optional_finite_time(value: Any, name: str = "time") -> Optional[float]:
    """Ensure value is None or a finite real number."""
    if value is None:
        return None
    return validate_finite_time(value, name)


def validate_nonboolean_int(
    value: Any,
    name: str = "value",
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Ensure value is an exact integer (not a bool) bounded by min/max if specified."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")
    return value


def validate_dependencies(
    deps: Sequence[Any],
    current_job_id: Optional[str] = None,
    name: str = "dependencies",
) -> Tuple[str, ...]:
    """Ensure dependencies are unique, non-self safe IDs."""
    if not isinstance(deps, (list, tuple, set, frozenset)):
        raise ValidationError(f"{name} must be a sequence or set, got {type(deps).__name__}")

    seen: Set[str] = set()
    cleaned: List[str] = []
    for idx, d in enumerate(deps):
        dep_id = validate_safe_id(d, f"{name}[{idx}]")
        if current_job_id is not None and dep_id == current_job_id:
            raise DependencyError(f"Job {current_job_id!r} cannot declare a dependency on itself")
        if dep_id in seen:
            raise DependencyError(f"Duplicate dependency detected in {name}: {dep_id!r}")
        seen.add(dep_id)
        cleaned.append(dep_id)
    return tuple(cleaned)


def validate_json_payload(
    payload: Any,
    name: str = "payload",
    *,
    max_depth: int = 16,
    max_bytes: int = 65536,
) -> Any:
    """Ensure payload is JSON-serializable, recursive depth <= 16, and serialized UTF-8 bytes <= 65536."""
    def _check_depth(obj: Any, current_depth: int) -> None:
        if current_depth > max_depth:
            raise ValidationError(
                f"{name} exceeds maximum recursion depth of {max_depth}"
            )
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise ValidationError(f"{name} dictionary key must be a string, got {type(k).__name__}")
                _check_depth(v, current_depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _check_depth(item, current_depth + 1)

    _check_depth(payload, 1)

    try:
        encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as err:
        raise ValidationError(f"{name} must be strictly JSON-serializable: {err}") from err

    if len(encoded) > max_bytes:
        raise ValidationError(
            f"{name} UTF-8 serialized size ({len(encoded)} bytes) exceeds maximum limit of {max_bytes} bytes"
        )

    return payload


# =====================================================================
# Frozen Dataclasses
# =====================================================================

@dataclass(frozen=True)
class RetryPolicy:
    """Policy defining retry bounds and backoff semantics."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    backoff_factor: float = 2.0
    jitter: bool = False

    def __post_init__(self) -> None:
        validate_nonboolean_int(self.max_retries, "max_retries", min_value=0)
        base = validate_finite_time(self.base_delay_seconds, "base_delay_seconds")
        if base < 0.0:
            raise ValidationError("base_delay_seconds must be non-negative")
        max_d = validate_finite_time(self.max_delay_seconds, "max_delay_seconds")
        if max_d < base:
            raise ValidationError("max_delay_seconds must be >= base_delay_seconds")
        factor = validate_finite_time(self.backoff_factor, "backoff_factor")
        if factor < 1.0:
            raise ValidationError("backoff_factor must be >= 1.0")
        if not isinstance(self.jitter, bool):
            raise ValidationError(f"jitter must be a boolean, got {type(self.jitter).__name__}")


@dataclass(frozen=True)
class JobSpec:
    """Immutable definition of a scheduled unit of work."""
    job_id: str
    task_type: str
    payload: Any = field(default_factory=dict)
    priority: int = 0
    timeout_seconds: float = 300.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    dependencies: Tuple[str, ...] = field(default_factory=tuple)
    created_at: float = 0.0

    def __post_init__(self) -> None:
        jid = validate_safe_id(self.job_id, "job_id")
        object.__setattr__(self, "job_id", jid)
        ttype = validate_safe_id(self.task_type, "task_type")
        object.__setattr__(self, "task_type", ttype)

        validate_nonboolean_int(self.priority, "priority", min_value=0)
        timeout = validate_finite_time(self.timeout_seconds, "timeout_seconds")
        if timeout <= 0.0:
            raise ValidationError("timeout_seconds must be positive")
        validate_finite_time(self.created_at, "created_at")

        if not isinstance(self.retry_policy, RetryPolicy):
            raise ValidationError(f"retry_policy must be a RetryPolicy instance, got {type(self.retry_policy).__name__}")

        validate_json_payload(self.payload, "payload")
        deps = validate_dependencies(self.dependencies, current_job_id=jid, name="dependencies")
        object.__setattr__(self, "dependencies", deps)


@dataclass(frozen=True)
class JobSnapshot:
    """Point-in-time state of a job in the lifecycle."""
    job_id: str
    state: JobState
    spec: JobSpec
    attempt: int = 0
    current_lease_id: Optional[str] = None
    worker_id: Optional[str] = None
    lease_expires_at: Optional[float] = None
    next_run_at: Optional[float] = None
    updated_at: float = 0.0
    last_error: Optional[str] = None
    failure_kind: Optional[FailureKind] = None

    def __post_init__(self) -> None:
        jid = validate_safe_id(self.job_id, "job_id")
        object.__setattr__(self, "job_id", jid)

        if not isinstance(self.state, JobState):
            try:
                object.__setattr__(self, "state", JobState(self.state))
            except ValueError as err:
                raise ValidationError(f"Invalid JobState: {self.state!r}") from err

        if not isinstance(self.spec, JobSpec):
            raise ValidationError(f"spec must be a JobSpec, got {type(self.spec).__name__}")
        if self.spec.job_id != jid:
            raise ValidationError(f"JobSnapshot job_id {jid!r} does not match spec.job_id {self.spec.job_id!r}")

        validate_nonboolean_int(self.attempt, "attempt", min_value=0)
        lid = validate_optional_safe_id(self.current_lease_id, "current_lease_id")
        object.__setattr__(self, "current_lease_id", lid)
        wid = validate_optional_safe_id(self.worker_id, "worker_id")
        object.__setattr__(self, "worker_id", wid)

        validate_optional_finite_time(self.lease_expires_at, "lease_expires_at")
        validate_optional_finite_time(self.next_run_at, "next_run_at")
        validate_finite_time(self.updated_at, "updated_at")

        if self.last_error is not None and not isinstance(self.last_error, str):
            raise ValidationError("last_error must be a string or None")

        if self.failure_kind is not None and not isinstance(self.failure_kind, FailureKind):
            try:
                object.__setattr__(self, "failure_kind", FailureKind(self.failure_kind))
            except ValueError as err:
                raise ValidationError(f"Invalid FailureKind: {self.failure_kind!r}") from err


@dataclass(frozen=True)
class LeaseClaim:
    """Lease token acquired by an active worker."""
    lease_id: str
    job_id: str
    worker_id: str
    acquired_at: float
    expires_at: float
    version: int = 1

    def __post_init__(self) -> None:
        lid = validate_safe_id(self.lease_id, "lease_id")
        object.__setattr__(self, "lease_id", lid)
        jid = validate_safe_id(self.job_id, "job_id")
        object.__setattr__(self, "job_id", jid)
        wid = validate_safe_id(self.worker_id, "worker_id")
        object.__setattr__(self, "worker_id", wid)

        acq = validate_finite_time(self.acquired_at, "acquired_at")
        exp = validate_finite_time(self.expires_at, "expires_at")
        if exp < acq:
            raise ValidationError(f"expires_at ({exp}) cannot precede acquired_at ({acq})")

        validate_nonboolean_int(self.version, "version", min_value=1)


@dataclass(frozen=True)
class EventRecord:
    """Immutable audit trail record for state transitions."""
    event_id: str
    job_id: str
    event_type: EventType
    timestamp: float
    worker_id: Optional[str] = None
    lease_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        eid = validate_safe_id(self.event_id, "event_id")
        object.__setattr__(self, "event_id", eid)
        jid = validate_safe_id(self.job_id, "job_id")
        object.__setattr__(self, "job_id", jid)

        if not isinstance(self.event_type, EventType):
            try:
                object.__setattr__(self, "event_type", EventType(self.event_type))
            except ValueError as err:
                raise ValidationError(f"Invalid EventType: {self.event_type!r}") from err

        validate_finite_time(self.timestamp, "timestamp")
        wid = validate_optional_safe_id(self.worker_id, "worker_id")
        object.__setattr__(self, "worker_id", wid)
        lid = validate_optional_safe_id(self.lease_id, "lease_id")
        object.__setattr__(self, "lease_id", lid)

        if self.details is not None:
            validate_json_payload(self.details, "details")


@dataclass(frozen=True)
class ConsistencyReport:
    """Report generated by storage and scheduler consistency verification."""
    is_consistent: bool
    checked_at: float
    total_jobs: int
    violations: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.is_consistent, bool):
            raise ValidationError(f"is_consistent must be a boolean, got {type(self.is_consistent).__name__}")
        validate_finite_time(self.checked_at, "checked_at")
        validate_nonboolean_int(self.total_jobs, "total_jobs", min_value=0)

        if not isinstance(self.violations, (list, tuple)):
            raise ValidationError("violations must be a sequence of strings")
        v_tuple = tuple(str(v) for v in self.violations)
        object.__setattr__(self, "violations", v_tuple)

        if self.metadata is not None:
            validate_json_payload(self.metadata, "metadata")