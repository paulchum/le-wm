import json
import tempfile
import unittest
from pathlib import Path

from drone_ai import (
    DroneActionCandidate,
    DroneObservation,
    MissionLog,
    OperatorInput,
    SafetyDecision,
    verify_mission_log,
)


class MissionLogTest(unittest.TestCase):
    def test_append_and_verify_hash_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mission.jsonl"
            log = MissionLog(path)
            observation = DroneObservation(timestamp_s=1.0, telemetry={"battery_percent": 90})
            proposed = DroneActionCandidate(1, 0, 0, 0, 1)
            decision = SafetyDecision(accepted=True, action=proposed)
            operator = OperatorInput("operator-1", "approve", True)

            entry = log.append(
                observation,
                proposed,
                decision,
                operator_input=operator,
                executed_action=proposed,
                outcome={"status": "executed"},
            )

            self.assertEqual(entry.sequence_id, 0)
            self.assertEqual(log.verify(), [])

    def test_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mission.jsonl"
            log = MissionLog(path)
            proposed = DroneActionCandidate(1, 0, 0, 0, 1)
            log.append(
                DroneObservation(timestamp_s=1.0),
                proposed,
                SafetyDecision(accepted=True, action=proposed),
            )

            record = json.loads(path.read_text(encoding="utf-8").strip())
            record["outcome"] = {"status": "tampered"}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            self.assertTrue(verify_mission_log(path))


if __name__ == "__main__":
    unittest.main()
