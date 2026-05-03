# Racer Helper — Human-Achievable Personalized Ghost for Trackmania

## Overview
**Racer Helper** gives competitive drivers a **personalized, realistic target** instead of an unrealistic “perfect lap.”  
It helps drivers improve faster by showing the **best line they can actually execute**, not the best line a robot could execute.

The system is designed for **Trackmania** and produces a driver-specific “best version” ghost plus actionable feedback.

---

## 1) Problem Definition

### What is the real-world problem being solved?
Professional and competitive drivers want to improve lap time, but each driver has a different:
- driving style (aggressiveness, smoothness, braking behavior),
- execution limits (reaction time, consistency),
- risk tolerance (how close they are willing to drive to the limit).

Existing “ideal” lines or perfect ghost laps are often unrealistic because they assume robot-like execution:
perfect braking points, perfect steering, no fatigue, and no reaction delay.

### What makes it a real problem?
In practice, drivers:
- miss braking points by milliseconds,
- oversteer/understeer,
- react differently under pressure,
- cannot consistently execute a single theoretical optimum.

This creates a real performance gap between:
- the **theoretical best lap**, and
- the **best lap a specific human driver can realistically achieve**.

Racer Helper addresses this gap by producing the **best realistic lap time and line** for a specific driver, conditioned on that driver’s style and risk preference.

---

## 2) Who Would Deploy / Pay for This System?

### Primary customer (initial target)
- **Competitive Trackmania drivers**

### Potential stakeholders (later expansion)
- real drivers and racing coaches (concept extension),
- karting training centers (future adaptation),
- esports racers and sim-racing players.

### Value proposition
- faster improvement,
- reduced training time,
- personalized feedback (where the driver loses time and what to improve).

---

## 3) Proposed Solution

Racer Helper generates a driver’s **“best version”** on a specific Trackmania map and outputs:
- a **realistic achievable lap time** (target),
- a **personalized ghost driver**,
- **segment-level guidance** on braking / turning / risk.

**Core insight:** optimal control should be customized to what the human driver can actually execute, not an ideal robot driver.

---

## 4) Method Overview

### Step 1 — Train RL under driver-conditioned noise
A reinforcement learning (RL) agent is trained in simulation while injecting **driver-specific execution noise** into actions.  
This forces the agent to learn the **fastest human-achievable policy** for that driver, rather than an unrealistically perfect policy.

### Step 2 — Output achievable time + personalized ghost
The system returns:
- a realistic lap target, and
- a ghost replay representing the driver’s best possible version on that map.

**Adaptive improvement loop:** as the driver improves and reaches the target, the system produces a faster target and a refined line based on the driver’s updated characteristics.

---

## 5) Baseline Strategy and ML Approach

### Non-AI Baseline (Mandatory)
A traditional deterministic racing-line heuristic:
- **Outside–Inside–Outside** cornering philosophy (classical racing line),
- optional simple braking-point rules based on track geometry/curvature.

This baseline provides a clear non-ML comparison for:
- line shape, and
- lap timing.

### ML Approach
- **Model family:** Reinforcement Learning (RL), conditioned on driving style and risk mode.

**Why RL is appropriate (beyond accuracy):**
- driving is a sequential control problem with long-term consequences,
- locally good actions can harm later corners (long-horizon trade-off),
- RL can optimize lap time while respecting action constraints and driver-specific execution limits.

---

## 6) Evaluation Plan

### Main comparisons
1. Non-AI baseline (heuristic line) vs RL-based personalized line.
2. Driver’s current laps vs personalized ghost “best version.”
3. Driver’s old line vs best-fitting personalized line, including realistic time improvement.

---

## 7) Deployment Plan
- Training runs on rented GPUs (cloud).
- Execution and visualization are performed locally.

---

## 8) Assumptions and Limitations

### Assumptions (project scope)
- Trackmania is treated as the operating environment / ground truth for this project.
- No weather variation, tire degradation, fuel limits, or random mechanical failures.

### Limitations
- retraining or adaptation is required for each track,
- the current system is limited to Trackmania due to tooling/interface dependency.

---

## 9) Responsible ML

- **Explainability / interpretability:** segment-level feedback (where time is lost and why).
- **Robustness:** performance comparison across different risk modes and driving styles.
- **Privacy / data leakage:** careful handling of driver telemetry and stored runs.

---

---

## 10) Driver Braking Aggression Conditioning

The system conditions the RL agent on a **braking aggression** scalar `α ∈ [0, 1]` representing the target driver's characteristic brake usage rate.

### How it works

| Component | Mechanism |
|---|---|
| **State input** | `α` is appended to the float feature vector (index 184). The network produces different Q-values for the same physical state depending on `α` — a Universal Value Function Approximator (UVFA) architecture. |
| **Reward signal** | A **Brier-score penalty** is added at each step: `r = coeff × (brake_t − α)²`. The Brier score is the unique *proper scoring rule* for binary events — its expectation is minimised exactly when the agent's empirical brake frequency equals `α`. |
| **Loss function** | The IQN quantile Huber loss is unchanged. The penalty flows through the Bellman targets: high-aggression profiles get higher target Q-values for brake actions; low-aggression profiles penalise unnecessary braking. |

### Configuration

In `config_files/config.py`:
```python
braking_aggression = 0.3          # target driver braking frequency [0, 1]
humanlike_braking_aggression_reward_schedule = [(0, -0.05)]
```

See §24 of the linesight README for the full mathematical derivation.

---

---

## 11) Oversteer / Understeer Style Conditioning

The system conditions the RL agent on an **oversteer/understeer score** `s ∈ [−5, 5]` representing the target driver's characteristic cornering style. Negative values describe an understeer-preferring driver (car pushes wide, front washes out); positive values describe an oversteer-preferring driver (rear steps out, countersteer corrections).

### Signal Construction

Two mutually exclusive per-step signals are derived from the car's velocity state:

| Signal | Condition | Value |
|---|---|---|
| **Oversteer** | lateral slip ratio `\|v_lat\| / \|v_fwd\|` is high | `clip(slip / 0.3, 0, 1) → [0, 1]` |
| **Understeer** | steering pressed AND slip ratio `< 0.04` | `1.0` (binary flag) |

These are combined into a single signed signal:

```
o_signal = clip(oversteer_component − understeer_component, −1, 1)
```

- `o = +1`: full oversteer (lateral slip saturated)
- `o = −1`: full understeer (steering applied, near-zero lateral response)
- `o =  0`: neutral tracking

The signal is suppressed below 10 m/s to avoid low-speed noise.

### Reward

```
r = coeff × (s / 5) × o_signal
```

The reward is **positive when the agent's cornering style aligns with the driver's score** and negative when misaligned:

| Score | Behavior | Reward |
|---|---|---|
| `s > 0` (oversteer driver) | agent oversteers (`o > 0`) | positive |
| `s < 0` (understeer driver) | agent understeers (`o < 0`) | positive |
| any | misaligned style | negative |
| `s = 0` | disabled | 0 |

### Configuration

In `config_files/config.py`:
```python
oversteer_understeer_score = 0.0   # [-5, 5]; 0 = disabled
humanlike_oversteer_understeer_reward_schedule = [(0, 0.05)]
```

Slip thresholds can be tuned in `trackmania_rl/buffer_management.py`:
```python
_OVERSTEER_SLIP_SAT  = 0.3   # |v_lat|/|v_fwd| at which oversteer signal saturates
_UNDERSTEER_SLIP_MAX = 0.04  # |v_lat|/|v_fwd| below which understeer fires when steering
```

---

---

## 12) Brake Tap Penalty

A brake press held for fewer than **3 consecutive steps (150 ms)** is treated as a micro-tap — no human driver commits to braking that briefly in a racing context. The violation is detected in a pre-pass over the rollout and a fixed penalty is applied at the release step.

```python
humanlike_brake_tap_penalty_schedule = [(0, -0.05)]
```

---

## 13) Steering & Accelerator Tap Penalties

Similar to the brake tap penalty but applied independently to **left/right steering** and **throttle** inputs. A press held fewer than 3 steps (150 ms) before release is penalised at the release step.

Left and right holds are tracked independently so a direct L→R transition (no neutral frame between) correctly tags the left release at the step right begins. The steer tap penalty is orthogonal to the steering oscillation penalty (§ oscillation): oscillation catches rapid direction alternation by frequency; tap catches any individual press that is too short regardless of direction.

```python
humanlike_steer_tap_penalty_schedule = [(0, -0.05)]
humanlike_accel_tap_penalty_schedule = [(0, -0.05)]
```

---

## 14) Corner Entry Speed Conditioning

The system conditions the driver profile on **corner entry speed** through a sparse reward term, using a target ratio `ρ ∈ [0, 1]` where:

```
ρ = median corner-entry speed / peak speed
```

### How it works

| Component | Mechanism |
|---|---|
| **Target profile** | `corner_entry_speed_ratio` represents how much speed the target driver carries into corners. Higher values mean later braking / faster corner entry. |
| **Corner detection** | A corner entry is detected when the curvature proxy `κ = \|yaw_rate_y\| / max(\|v_fwd\|, 1.0)` crosses the `0.010 1/m` threshold. |
| **Reward signal** | At each detected entry, a penalty is applied: `r = coeff × (entry_ratio − ρ)²`. Since `coeff` is negative, mismatched entry speed is penalized. |

This conditioning is applied during buffer construction; it does not add a new network input.

### Configuration

In `config_files/config.py`:
```python
corner_entry_speed_ratio = 0.84
humanlike_corner_entry_speed_reward_schedule = [(0, -0.2)]
```

The ratio can be copied from `extract_driver_profile.py` after running it on a driver replay.

---

## 15) Extract Driver Profile Script

`linesight/scripts/extract_driver_profile.py` converts a Trackmania `.Replay.Gbx` into the driver-style values used by the conditioning system.

Run from `linesight/`:
```bash
PYTHONPATH=. python scripts/extract_driver_profile.py <replay.Replay.Gbx>
```

### Outputs

| Output | Purpose |
|---|---|
| `braking_aggression` | Target brake usage profile in `[0, 1]` |
| `oversteer_understeer_score` | Cornering style score in `[-5, 5]` |
| `corner_entry_speed_level` | Coarse entry-speed label: `low`, `medium`, or `high` |
| `corner_entry_speed_ratio` | Numeric target used by the corner-entry-speed reward |

The script also prints the config values to paste into `config_files/config.py` before training or evaluating a personalized driver profile.

---

## Team
- Jad Al Lakkis  
- Ibrahim Khaled
