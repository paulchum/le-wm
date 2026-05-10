# Canadian Defence Full-Stack Drone Autonomy

This repository is a drone AI autonomy-kit codebase that uses LeWorldModel as a
learned world-model component. The public `drone_ai` package supports a Canadian
defence R&D program with two offerings:

- **Autonomy Kit**: software, edge compute integration, sensors, ROS/PX4 or
  ROS/ArduPilot integration, operator UI hooks, logs, and validation artifacts
  that can be installed on partner airframes.
- **Canadian Reference Drone Platform**: a complete UAS prototype that packages
  the autonomy kit with a reference airframe, payloads, ground station, secure
  data pipeline, support model, and qualification evidence.

The first product should be the Autonomy Kit exposed by the `drone_ai` package.
The reference platform should be developed in parallel as an integration target,
not as the first dependency for market entry.

Kinetic mission contexts are represented only as operating context for mobility
autonomy. This package does not implement autonomous target selection, payload
release, weapon release, or lethal effect decisions.

## Runtime Boundary

LeWM is the learned world-model component. It can score short-horizon action
candidates, predict latent rollouts, and emit surprise/anomaly scores. It must
not directly command motors, bypass a flight controller, bypass the operator, or
make kinetic decisions.

The intended runtime chain is:

1. Sensors and flight controller publish an observation.
2. `VelocityLatticeSampler` or mission logic generates candidate movement
   setpoints.
3. `SafetySupervisor` pre-filters each candidate.
4. `WorldModelPlanner` scores accepted candidates with LeWM or a bench-test
   heuristic planner.
5. `AutonomyKit` recommends the best accepted movement setpoint.
6. In `dry_run` or `closed_loop` mode, PX4 or ArduPilot receives only
   safety-approved movement setpoints through a `FlightController` adapter.
7. `MissionLog` records the observation, proposed action, safety decision,
   operator input, executed action, and outcome.

## Public Interfaces

The package exposes these integration contracts:

- `drone_ai.DroneObservation`: timestamped pixels, telemetry, proprioception, mission
  state, environment metadata, and optional model tensors.
- `drone_ai.DroneActionCandidate`: body-frame velocity, yaw-rate, vertical-rate or
  altitude setpoint, duration, source, and safety metadata.
- `drone_ai.WorldModelPlanner`: `score_action_candidates`, `rollout`, and `surprise`.
- `drone_ai.SafetySupervisor`: accepts, rejects, or clips candidate actions against a
  `SafetyEnvelope`.
- `drone_ai.MissionLog`: append-only JSONL records with hash chaining for auditability.
- `drone_ai.AutonomyKit`: orchestration loop for candidate generation, safety filtering,
  world-model scoring, optional flight-controller handoff, and mission logging.
- `drone_ai.DryRunFlightController`: local adapter for simulation and bench validation.

## Canadian Defence Posture

Default assumptions for this integration layer:

- Non-kinetic use cases only: ISR support, autonomy assurance, defence
  infrastructure inspection, logistics autonomy research, navigation research,
  and counter-UxS test/evaluation support.
- Kinetic-support contexts may be configured for mobility-only tasks such as
  repositioning, route recommendation, overwatch viewpoint selection, post-effect
  assessment routing, and avoiding hazardous areas. Effects authority remains
  outside this package.
- Candidate metadata that attempts `effect_command`, `kinetic_effect`,
  `payload_release`, `weapon_release`, or `target_engagement` is rejected by the
  safety supervisor.
- Canadian-controlled IP and Canadian-hosted logs/model artifacts by default.
- Controlled Goods, export-control, cybersecurity, and secure-facility reviews
  happen before handling sensitive payloads, classified requirements, controlled
  technical data, or exportable defence articles.
- NRC Drone Innovation Hub, DRDC, DND end-user trials, IDEaS, and ITB-driven
  prime partnerships are the preferred validation and commercialization paths.

## Acceptance Gates

Use these gates before any fielded demonstration:

1. Reproduce existing LeWM benchmark behavior.
2. Train and evaluate simulated drone datasets.
3. Run passive onboard logging with no learned commands executed.
4. Run advisory-only recommendations with operator review.
5. Run safety-filtered closed-loop flight in controlled conditions.
6. Run partner-airframe trials.
7. Package the reference platform prototype.

Track goal-reaching rate, safety-filter rejection rate, prediction error,
anomaly precision/recall, p95 edge latency, operator overrides, and repeatability
across environments.

## Reference Platform V1

The selected reference-platform baseline is a small VTOL RPAS under 25 kg. That
keeps the prototype within the small-RPAS evidence frame while giving the
platform a credible BVLOS endurance story. Multirotor airframes remain useful
for early autonomy-kit bench work, but the reference UAS should exercise hover,
transition, fixed-wing cruise, C2, lost-link, payload, and recovery assumptions.

The v1 payload architecture is configurable, with EO/IR and depth/range modules
as the default evidence profile. Payloads must declare mass, power, interfaces,
data products, autonomy roles, classification ceiling, and non-kinetic posture.
The reference manifest rejects payloads that exceed bay limits, require blocked
classification labels, or violate the non-kinetic constraint.

The React/Vite ground station is an unclassified operator-interface scaffold. It
shows route/containment state, aircraft health, payload health, autonomy
recommendations, secure pipeline status, evidence readiness, and hash-chain log
verification. It is not a certified control station, but it creates the concrete
surface needed for human-factors review and Standard 922 evidence planning.

The secure data pipeline hashes artifacts, emits review-gated bundle manifests,
records encryption/KMS expectations, and blocks `classified`,
`controlled_goods`, `secret`, and `top_secret` labels. Classified payload data
and controlled technical data must remain outside this repository and move only
through approved program facilities, networks, and export-control/Controlled
Goods review.

The BVLOS/L1C evidence scaffold tracks:

- Standard 922.08 containment and operational volume evidence
- Standard 922.09 command-and-control link and lost-link behavior
- Standard 922.10 detect/alert/avoid assumptions
- Standard 922.11 control station design and operator awareness
- Standard 922.12 environmental envelope and reliability limits
- program security gates for classified-path handling
- non-kinetic autonomy assurance from the safety supervisor and mission log

## Developer Smoke Test

The source-tree example exercises the kit without a LeWM checkpoint:

```bash
python3 examples/canadian_defence_autonomy_kit.py
```

It uses `HeuristicWorldModelPlanner`, `SafetySupervisor`,
`DryRunFlightController`, and `MissionLog` to produce a full decision record.
