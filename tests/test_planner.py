import tempfile
import unittest
from pathlib import Path

from drone_ai import (
    AutonomyKit,
    AutonomyKitConfig,
    AutonomyMode,
    CandidateSamplerConfig,
    DroneActionCandidate,
    DroneObservation,
    DryRunFlightController,
    GeoRasterLayer,
    GeoWorldModelInput,
    HeuristicWorldModelPlanner,
    LeWMWorldModelPlanner,
    MissionEffectContext,
    MissionLog,
    OperatorInput,
    SafetyEnvelope,
    SafetySupervisor,
    VelocityLatticeSampler,
)


try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


class VelocityLatticeSamplerTest(unittest.TestCase):
    def test_goal_directed_candidate_is_generated(self):
        sampler = VelocityLatticeSampler(
            CandidateSamplerConfig(
                forward_speeds_mps=(0.0,),
                lateral_speeds_mps=(0.0,),
                vertical_speeds_mps=(0.0,),
                yaw_rates_rps=(0.0,),
                include_hover=False,
                include_goal_directed=True,
            )
        )

        candidates = sampler.generate(
            DroneObservation(timestamp_s=0, telemetry={"x_m": 0, "y_m": 0}),
            DroneObservation(timestamp_s=1, telemetry={"x_m": 10, "y_m": 0}),
        )

        self.assertTrue(any(candidate.source == "goal_directed" for candidate in candidates))


class AutonomyKitTest(unittest.TestCase):
    def test_advisory_mode_recommends_without_execution(self):
        kit = AutonomyKit(
            planner=HeuristicWorldModelPlanner(),
            candidate_sampler=VelocityLatticeSampler(
                CandidateSamplerConfig(
                    forward_speeds_mps=(0.0, 1.0),
                    lateral_speeds_mps=(0.0,),
                    vertical_speeds_mps=(0.0,),
                    yaw_rates_rps=(0.0,),
                    include_hover=False,
                    include_goal_directed=False,
                )
            ),
            flight_controller=DryRunFlightController(),
        )

        decision = kit.step(
            DroneObservation(
                timestamp_s=0,
                telemetry={"x_m": 0, "y_m": 0, "battery_percent": 90, "altitude_m": 10},
                mission_state={"operator_authority": True},
            ),
            DroneObservation(timestamp_s=1, telemetry={"x_m": 5, "y_m": 0}),
        )

        self.assertEqual(decision.status, "recommended")
        self.assertIsNotNone(decision.selected_action)
        self.assertIsNone(decision.executed_action)

    def test_dry_run_executes_after_operator_approval_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = DryRunFlightController()
            log = MissionLog(Path(tmpdir) / "mission.jsonl")
            kit = AutonomyKit(
                planner=HeuristicWorldModelPlanner(),
                flight_controller=controller,
                mission_log=log,
                config=AutonomyKitConfig(mode=AutonomyMode.DRY_RUN),
            )

            decision = kit.step(
                DroneObservation(
                    timestamp_s=0,
                    telemetry={"x_m": 0, "y_m": 0, "battery_percent": 90, "altitude_m": 10},
                    mission_state={"operator_authority": True},
                ),
                DroneObservation(timestamp_s=1, telemetry={"x_m": 5, "y_m": 0}),
                operator_input=None,
            )
            approved = kit.step(
                DroneObservation(
                    timestamp_s=2,
                    telemetry={"x_m": 0, "y_m": 0, "battery_percent": 90, "altitude_m": 10},
                    mission_state={"operator_authority": True},
                ),
                DroneObservation(timestamp_s=3, telemetry={"x_m": 5, "y_m": 0}),
                operator_input=OperatorInput("op", "approve", True),
            )

            self.assertEqual(decision.status, "operator_approval_required")
            self.assertEqual(approved.status, "executed")
            self.assertEqual(len(controller.commands), 1)
            self.assertEqual(log.verify(), [])

    def test_kinetic_support_context_does_not_enable_effect_commands(self):
        kit = AutonomyKit(
            planner=HeuristicWorldModelPlanner(),
            safety_supervisor=SafetySupervisor(
                SafetyEnvelope(require_non_kinetic=False, allow_kinetic_context=True)
            ),
            config=AutonomyKitConfig(effect_context=MissionEffectContext.KINETIC_SUPPORT),
        )
        observation = DroneObservation(
            timestamp_s=0,
            telemetry={"battery_percent": 90, "altitude_m": 10},
            mission_state={"operator_authority": True},
        )
        goal = DroneObservation(timestamp_s=1)

        context_decision = kit.step(
            observation,
            goal,
            candidates=[DroneActionCandidate(1, 0, 0, 0, 1)],
        )
        blocked_decision = kit.step(
            observation,
            goal,
            candidates=[DroneActionCandidate(1, 0, 0, 0, 1, metadata={"weapon_release": True})],
        )

        self.assertTrue(context_decision.accepted)
        self.assertFalse(blocked_decision.accepted)
        self.assertIn("effect_command_blocked", blocked_decision.safety_decision.reasons)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PlannerAdapterTest(unittest.TestCase):
    def test_scores_candidates_with_lewm_cost_api(self):
        class FakeModel:
            def eval(self):
                self.was_eval = True

            def get_cost(self, info, action_candidates):
                self.info = info
                return action_candidates.pow(2).sum(dim=(2, 3))

        model = FakeModel()
        planner = LeWMWorldModelPlanner(model)
        observation = DroneObservation(
            timestamp_s=1.0,
            model_inputs={"pixels": torch.zeros(1, 3, 3, 224, 224)},
        )
        goal = DroneObservation(
            timestamp_s=2.0,
            model_inputs={"pixels": torch.ones(1, 3, 3, 224, 224)},
        )

        costs = planner.score_action_candidates(
            observation,
            goal,
            [
                DroneActionCandidate(1, 0, 0, 0, 1),
                DroneActionCandidate(0, 2, 0, 0, 1),
            ],
        )

        self.assertEqual(costs, [5.0, 20.0])
        self.assertTrue(model.was_eval)
        self.assertIn("goal", model.info)

    def test_pads_drone_actions_to_checkpoint_action_width(self):
        class FakeActionEncoder:
            class PatchEmbed:
                in_channels = 6

            patch_embed = PatchEmbed()

        class FakeModel:
            action_encoder = FakeActionEncoder()

            def eval(self):
                pass

            def get_cost(self, info, action_candidates):
                self.action_shape = tuple(action_candidates.shape)
                self.first_action = action_candidates[0, 0, 0].detach().cpu().tolist()
                return action_candidates.sum(dim=(2, 3))

        model = FakeModel()
        planner = LeWMWorldModelPlanner(model)
        observation = DroneObservation(
            timestamp_s=1.0,
            model_inputs={"pixels": torch.zeros(1, 3, 3, 224, 224)},
        )
        goal = DroneObservation(
            timestamp_s=2.0,
            model_inputs={"pixels": torch.ones(1, 3, 3, 224, 224)},
        )

        planner.score_action_candidates(
            observation,
            goal,
            [DroneActionCandidate(1, 2, 3, 4, 1)],
        )

        self.assertEqual(model.action_shape, (1, 1, 5, 6))
        self.assertEqual(model.first_action, [1.0, 2.0, 3.0, 4.0, 0.0, 0.0])

    def test_converts_geo_world_model_inputs_before_scoring(self):
        class FakeModel:
            def eval(self):
                pass

            def get_cost(self, info, action_candidates):
                self.info = info
                return action_candidates.sum(dim=(2, 3))

        model = FakeModel()
        planner = LeWMWorldModelPlanner(model)
        observation = DroneObservation(
            timestamp_s=1.0,
            model_inputs=GeoWorldModelInput(
                pixels=torch.zeros(1, 3, 3, 8, 8),
                dynamic_rasters={
                    "weather": GeoRasterLayer(torch.ones(1, 3, 1, 8, 8))
                },
            ),
        )
        goal = DroneObservation(
            timestamp_s=2.0,
            model_inputs=GeoWorldModelInput(pixels=torch.ones(1, 3, 3, 8, 8)),
        )

        planner.score_action_candidates(
            observation,
            goal,
            [DroneActionCandidate(1, 0, 0, 0, 1)],
        )

        self.assertIn("geo_dynamic_weather", model.info)
        self.assertTrue(model.info["geo_present_dynamic_weather"].all())
        self.assertIn("geo_dynamic_ice", model.info)
        self.assertFalse(model.info["geo_present_dynamic_ice"].any())
        self.assertIn("geo_imagery_orthomosaic", model.info)
        self.assertFalse(model.info["geo_present_imagery_orthomosaic"].any())

    def test_empty_candidate_list_returns_empty_scores(self):
        planner = LeWMWorldModelPlanner(object())

        self.assertEqual(
            planner.score_action_candidates(
                DroneObservation(timestamp_s=1),
                DroneObservation(timestamp_s=2),
                [],
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
