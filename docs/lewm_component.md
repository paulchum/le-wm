# LeWM Component Boundary

This repo is a drone AI autonomy-kit repo. LeWorldModel is one component inside
that stack.

## Role

LeWM is responsible for learned world-model reasoning:

- encode image observations into compact latent embeddings
- predict next latent state from recent embeddings and action candidates
- score short-horizon movement candidates against a goal
- estimate surprise when the next observation diverges from prediction

The drone-facing adapter is `drone_ai.LeWMWorldModelPlanner`.

## Source Files

- `jepa.py`: LeWM model wrapper with `encode`, `predict`, `rollout`, and
  `get_cost`
- `module.py`: transformer predictor, action embedder, MLPs, and SIGReg
- `train.py`: LeWM training entrypoint
- `eval.py`: benchmark/evaluation entrypoint

## Not Owned By LeWM

LeWM does not own:

- flight stabilization
- direct motor or actuator control
- geofence, altitude, battery, comms, or operator-authority checks
- payload release, weapon release, target engagement, or effects authority
- mission logging or deployment approval

Those responsibilities belong to the drone AI stack around it: `AutonomyKit`,
`SafetySupervisor`, `FlightController`, operator tooling, and mission governance.

## Bench Mode

Use `drone_ai.HeuristicWorldModelPlanner` for local kit tests when a LeWM
checkpoint is unavailable. It preserves the same planner interface but does not
replace LeWM for real learned autonomy work.
