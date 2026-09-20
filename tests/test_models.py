"""Unit test suite for joblab.models.

Verifies domain enums, validation routines, JSON payload constraints,
retry policy bounds, job specification invariants, snapshots, leases,
event records, and consistency reporting.
"""

from __future__ import annotations

import math
import unittest

from joblab.models import (
    ConflictError,
    ConsistencyReport,
    CorruptionError,
    DependencyError,
    EventRecord,
    EventType,
    FailureKind,
    JobLabError,
    JobSnapshot,
    JobSpec,
    JobState,
    LeaseClaim,
    LeaseError,
    RetryPolicy,
    ValidationError,
    validate_dependencies,
    validate_finite_time,
    validate_nonboolean_int,
    validate_optional_finite_time,
    validate_optional_safe_id,
    validate_safe_id,
)


class TestModels(unittest.TestCase):
    """Behavioral unit tests for domain models and validation rules."""

    def test_domain_enums(self) -> None:
        """Verify domain enums contain valid members and expected base types."""
        self.assertGreater(len(JobState), 0)
        self.assertGreater(len(EventType), 0)
        self.assertGreater(len(FailureKind), 0)
        for state in JobState:
            self.assertIsInstance(state.value, str)
        for event_type in EventType:
            self.assertIsInstance(event_type.value, str)
        for kind in FailureKind:
            self.assertIsInstance(kind.value, str)

    def test_validate_safe_id(self) -> None:
        """Verify safe ID validation accepts conformant IDs and rejects malformed values."""
        self.assertEqual(validate_safe_id("job-001_alpha", "id"), "job-001_alpha")
        with self.assertRaises(ValidationError):
            validate_safe_id("", "id")
        with self.assertRaises(ValidationError):
            validate_safe_id("id with spaces", "id")
        with self.assertRaises(ValidationError):
            validate_safe_id("invalid/character", "id")
        with self.assertRaises(ValidationError):
            validate_safe_id(12345, "id")

    def test_validate_optional_safe_id(self) -> None:
        """Verify optional safe ID accepts None or conformant IDs, rejecting malformed values."""
        self.assertIsNone(validate_optional_safe_id(None, "opt_id"))
        self.assertEqual(validate_optional_safe_id("lease-xyz", "opt_id"), "lease-xyz")
        with self.assertRaises(ValidationError):
            validate_optional_safe_id("bad id!", "opt_id")

    def test_validate_finite_time(self) -> None:
        """Verify finite timestamp validation rejects non-numbers, booleans, and non-finite floats."""
        self.assertEqual(validate_finite_time(100.5, "ts"), 100.5)
        self.assertEqual(validate_finite_time(0, "ts"), 0.0)
        with self.assertRaises(ValidationError):
            validate_finite_time(float("nan"), "ts")
        with self.assertRaises(ValidationError):
            validate_finite_time(float("inf"), "ts")
        with self.assertRaises(ValidationError):
            validate_finite_time(float("-inf"), "ts")
        with self.assertRaises(ValidationError):
            validate_finite_time(True, "ts")
        with self.assertRaises(ValidationError):
            validate_finite_time("100.5", "ts")

    def test_validate_optional_finite_time(self) -> None:
        """Verify optional finite timestamp allows None and validates finite float values."""
        self.assertIsNone(validate_optional_finite_time(None, "opt_ts"))
        self.assertEqual(validate_optional_finite_time(50.0, "opt_ts"), 50.0)
        with self.assertRaises(ValidationError):
            validate_optional_finite_time(float("nan"), "opt_ts")

    def test_validate_nonboolean_int(self) -> None:
        """Verify bounded integer validation rejects booleans, floats, and strings."""
        self.assertEqual(validate_nonboolean_int(10, "val"), 10)
        self.assertEqual(validate_nonboolean_int(0, "val"), 0)
        with self.assertRaises(ValidationError):
            validate_nonboolean_int(True, "val")
        with self.assertRaises(ValidationError):
            validate_nonboolean_int(False, "val")
        with self.assertRaises(ValidationError):
            validate_nonboolean_int(10.5, "val")
        with self.assertRaises(ValidationError):
            validate_nonboolean_int("10", "val")

    def test_validate_dependencies(self) -> None:
        """Verify dependency validation normalizes iterables to tuples and rejects invalid items."""
        result = validate_dependencies(["job-a", "job-b"], "dependencies")
        self.assertEqual(result, ("job-a", "job-b"))
        self.assertIsInstance(result, tuple)
        with self.assertRaises(ValidationError):
            validate_dependencies(["invalid id with space"], "dependencies")
        with self.assertRaises(ValidationError):
            validate_dependencies("not-a-sequence", "dependencies")

    def test_json_payload_constraints(self) -> None:
        """Verify JSON payload validation rejects non-string keys and non-finite floats."""
        valid_spec = JobSpec(
            job_id="job-json-001",
            task_type="data-sync",
            payload={"metric": 12.5, "nested": {"active": True, "items": [1, 2]}},
        )
        self.assertIn("metric", valid_spec.payload)

        with self.assertRaises(ValidationError):
            JobSpec(
                job_id="job-json-bad-key",
                task_type="data-sync",
                payload={100: "non-string-key"},
            )
        with self.assertRaises(ValidationError):
            JobSpec(
                job_id="job-json-nan",
                task_type="data-sync",
                payload={"val": float("nan")},
            )
        with self.assertRaises(ValidationError):
            JobSpec(
                job_id="job-json-inf",
                task_type="data-sync",
                payload={"val": float("inf")},
            )

    def test_retry_policy_constraints(self) -> None:
        """Verify RetryPolicy invariants, default constructor, and rejection of invalid bounds."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=300.0,
            backoff_factor=2.0,
            jitter=False,
        )
        self.assertEqual(policy.max_retries, 3)
        self.assertEqual(policy.base_delay_seconds, 1.0)
        self.assertEqual(policy.max_delay_seconds, 300.0)
        self.assertEqual(policy.backoff_factor, 2.0)
        self.assertFalse(policy.jitter)

        with self.assertRaises(ValidationError):
            RetryPolicy(max_retries=-1)
        with self.assertRaises(ValidationError):
            RetryPolicy(base_delay_seconds=500.0, max_delay_seconds=300.0)
        zero_delay_policy = RetryPolicy(base_delay_seconds=0.0)
        self.assertEqual(zero_delay_policy.base_delay_seconds, 0.0)
        with self.assertRaises(ValidationError):
            RetryPolicy(base_delay_seconds=float("nan"))
        with self.assertRaises(ValidationError):
            RetryPolicy(backoff_factor=0.5)

    def test_job_spec_constraints_and_dependencies(self) -> None:
        """Verify JobSpec invariants, dependency normalization, and timeout checks."""
        spec = JobSpec(
            job_id="job-spec-001",
            task_type="render-frame",
            payload={},
            priority=5,
            timeout_seconds=60.0,
            retry_policy=RetryPolicy(),
            dependencies=["dep-parent-1", "dep-parent-2"],
            created_at=1000.0,
        )
        self.assertEqual(spec.job_id, "job-spec-001")
        self.assertEqual(spec.dependencies, ("dep-parent-1", "dep-parent-2"))
        self.assertIsInstance(spec.dependencies, tuple)

        with self.assertRaises(ValidationError):
            JobSpec(job_id="", task_type="render-frame")
        with self.assertRaises(ValidationError):
            JobSpec(job_id="job-spec-001", task_type="")
        with self.assertRaises(ValidationError):
            JobSpec(job_id="job-spec-001", task_type="render-frame", timeout_seconds=0.0)
        with self.assertRaises(ValidationError):
            JobSpec(job_id="job-spec-001", task_type="render-frame", timeout_seconds=-5.0)
        with self.assertRaises(ValidationError):
            JobSpec(job_id="job-spec-001", task_type="render-frame", timeout_seconds=float("nan"))

    def test_job_snapshot_mismatch(self) -> None:
        """Verify JobSnapshot enforces consistency between snapshot job_id and spec job_id."""
        spec = JobSpec(job_id="job-snap-001", task_type="train-model")
        state = list(JobState)[0]
        snapshot = JobSnapshot(job_id="job-snap-001", state=state, spec=spec)
        self.assertEqual(snapshot.job_id, spec.job_id)
        self.assertEqual(snapshot.spec.job_id, "job-snap-001")

        with self.assertRaises(ValidationError):
            JobSnapshot(job_id="job-snap-mismatch", state=state, spec=spec)

    def test_lease_claim_expiry(self) -> None:
        """Verify LeaseClaim permits expires_at == acquired_at and rejects expires_at < acquired_at."""
        equal_lease = LeaseClaim(
            lease_id="lease-001",
            job_id="job-001",
            worker_id="worker-001",
            acquired_at=500.0,
            expires_at=500.0,
            version=1,
        )
        self.assertEqual(equal_lease.acquired_at, 500.0)
        self.assertEqual(equal_lease.expires_at, 500.0)

        greater_lease = LeaseClaim(
            lease_id="lease-002",
            job_id="job-001",
            worker_id="worker-001",
            acquired_at=500.0,
            expires_at=530.0,
            version=1,
        )
        self.assertGreater(greater_lease.expires_at, greater_lease.acquired_at)

        with self.assertRaises(ValidationError):
            LeaseClaim(
                lease_id="lease-003",
                job_id="job-001",
                worker_id="worker-001",
                acquired_at=500.0,
                expires_at=499.0,
                version=1,
            )

    def test_event_record_validation(self) -> None:
        """Verify EventRecord accepts valid fields and rejects bad timestamps, event types, or IDs."""
        event_type = list(EventType)[0]
        event = EventRecord(
            event_id="evt-001",
            job_id="job-001",
            event_type=event_type,
            timestamp=1234.5,
            worker_id="worker-001",
            lease_id="lease-001",
            details={"action": "step-complete"},
        )
        self.assertEqual(event.event_id, "evt-001")
        self.assertEqual(event.timestamp, 1234.5)

        with self.assertRaises(ValidationError):
            EventRecord(
                event_id="evt-002",
                job_id="job-001",
                event_type=event_type,
                timestamp=float("nan"),
            )
        with self.assertRaises(ValidationError):
            EventRecord(
                event_id="evt-003",
                job_id="job-001",
                event_type="UNREGISTERED_EVENT_TYPE",
                timestamp=1234.5,
            )
        with self.assertRaises(ValidationError):
            EventRecord(
                event_id="",
                job_id="job-001",
                event_type=event_type,
                timestamp=1234.5,
            )

    def test_consistency_report_validation(self) -> None:
        """Verify ConsistencyReport checks types/metadata and does not enforce flag-violation correlation."""
        report = ConsistencyReport(
            is_consistent=True,
            checked_at=100.0,
            total_jobs=10,
            violations=("unreferenced_lease",),
            metadata={"collector": "cron"},
        )
        self.assertTrue(report.is_consistent)
        self.assertEqual(report.violations, ("unreferenced_lease",))
        self.assertEqual(report.metadata, {"collector": "cron"})

        with self.assertRaises(ValidationError):
            ConsistencyReport(
                is_consistent=True,
                checked_at=float("nan"),
                total_jobs=10,
            )
        with self.assertRaises(ValidationError):
            ConsistencyReport(
                is_consistent=True,
                checked_at=100.0,
                total_jobs=-1,
            )

    def test_exception_hierarchy(self) -> None:
        """Verify model error classes inherit from JobLabError."""
        self.assertTrue(issubclass(ValidationError, JobLabError))
        self.assertTrue(issubclass(ConflictError, JobLabError))
        self.assertTrue(issubclass(DependencyError, JobLabError))
        self.assertTrue(issubclass(LeaseError, JobLabError))
        self.assertTrue(issubclass(CorruptionError, JobLabError))


if __name__ == "__main__":
    unittest.main()
