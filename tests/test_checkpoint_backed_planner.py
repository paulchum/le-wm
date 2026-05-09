import math
import os
import unittest

from drone_ai import (
    AutonomyKit,
    AutonomyKitConfig,
    DroneActionCandidate,
    LEWM_HF_MODEL_REPOS,
    LeWMCheckpointSpec,
    load_hf_lewm_world_model_planner,
)
from examples.checkpoint_backed_drone_demo import _demo_observations


try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipUnless(
    os.environ.get("LEWM_RUN_CHECKPOINT_TEST") == "1",
    "set LEWM_RUN_CHECKPOINT_TEST=1 to download/load the HF LeWM checkpoint",
)
@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointBackedPlannerTest(unittest.TestCase):
    def test_autonomy_kit_scores_candidates_with_real_hf_checkpoint(self):
        repo_id = os.environ.get("LEWM_HF_REPO_ID", LEWM_HF_MODEL_REPOS["pusht"])
        device = os.environ.get("LEWM_DEVICE", "cpu")
        checkpoint_dir = os.environ.get("LEWM_CHECKPOINT_DIR")

        planner = load_hf_lewm_world_model_planner(
            LeWMCheckpointSpec(repo_id=repo_id, local_dir=checkpoint_dir),
            device=device,
        )
        observation, goal = _demo_observations(torch, device)
        candidates = [
            DroneActionCandidate(0.0, 0.0, 0.0, 0.0, 1.0),
            DroneActionCandidate(1.0, 0.0, 0.0, 0.0, 1.0),
            DroneActionCandidate(0.0, 1.0, 0.0, 0.15, 1.0),
        ]
        kit = AutonomyKit(
            planner=planner,
            config=AutonomyKitConfig(max_candidates=len(candidates), log_all_candidates=True),
        )

        with torch.no_grad():
            decision = kit.step(observation, goal, candidates=candidates)

        costs = [cost for cost in decision.costs if cost is not None]
        self.assertEqual(decision.status, "recommended")
        self.assertEqual(len(costs), len(candidates))
        self.assertTrue(all(math.isfinite(cost) for cost in costs))
        self.assertIsNotNone(decision.selected_action)


if __name__ == "__main__":
    unittest.main()
