"""Append-only mission logging for autonomy audit trails."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .interfaces import (
    DroneActionCandidate,
    DroneObservation,
    MissionLogEntry,
    OperatorInput,
    SafetyDecision,
    to_jsonable,
)

GENESIS_HASH = "0" * 64


class MissionLog:
    """JSONL mission log with hash chaining.

    Each record includes the previous record hash and its own canonical hash.
    This is not a cryptographic signing system, but it makes accidental edits
    and record removal visible during verification.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence_id, self._previous_hash = self._load_tail()

    def append(
        self,
        observation: DroneObservation,
        proposed_action: DroneActionCandidate,
        safety_decision: SafetyDecision,
        *,
        operator_input: OperatorInput | None = None,
        executed_action: DroneActionCandidate | None = None,
        outcome: Mapping[str, Any] | None = None,
    ) -> MissionLogEntry:
        sequence_id = self._sequence_id
        previous_hash = self._previous_hash
        record_without_hash = {
            "sequence_id": sequence_id,
            "created_at_s": time.time(),
            "observation": observation.summary(),
            "proposed_action": to_jsonable(proposed_action),
            "safety_decision": safety_decision.to_record(),
            "operator_input": to_jsonable(operator_input) if operator_input else None,
            "executed_action": to_jsonable(executed_action) if executed_action else None,
            "outcome": to_jsonable(outcome or {}),
            "previous_hash": previous_hash,
        }
        entry_hash = _hash_record(record_without_hash)
        record = {**record_without_hash, "entry_hash": entry_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

        self._sequence_id += 1
        self._previous_hash = entry_hash
        return MissionLogEntry(**record)

    def records(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def verify(self) -> list[str]:
        return verify_mission_log(self.path)

    def _load_tail(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS_HASH
        last_record: dict[str, Any] | None = None
        for record in self.records():
            last_record = record
        if last_record is None:
            return 0, GENESIS_HASH
        return int(last_record["sequence_id"]) + 1, str(last_record["entry_hash"])


def verify_mission_log(path: str | Path) -> list[str]:
    """Return verification errors for a mission log, or an empty list."""

    path = Path(path)
    if not path.exists():
        return [f"log_not_found:{path}"]

    errors: list[str] = []
    previous_hash = GENESIS_HASH
    expected_sequence = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line_{line_number}:invalid_json:{exc.msg}")
                continue

            if record.get("sequence_id") != expected_sequence:
                errors.append(f"line_{line_number}:sequence_mismatch")
            if record.get("previous_hash") != previous_hash:
                errors.append(f"line_{line_number}:previous_hash_mismatch")

            expected_hash = record.get("entry_hash")
            record_without_hash = dict(record)
            record_without_hash.pop("entry_hash", None)
            actual_hash = _hash_record(record_without_hash)
            if expected_hash != actual_hash:
                errors.append(f"line_{line_number}:entry_hash_mismatch")

            previous_hash = str(expected_hash)
            expected_sequence += 1
    return errors


def _hash_record(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
