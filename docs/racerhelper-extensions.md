# RacerHelper — Extensions & Driver Conditioning (Sections 23–25)

> What we added on top of Linesight to produce a **personalized, human-achievable ghost** for a specific driver.  
> Base model: [Linesight](https://github.com/linesight-rl/linesight) — see [linesight-technical.md](./linesight-technical.md) for core architecture.

---

## Table of Contents

23. [Human-Likeness Penalties](#23-human-likeness-penalties)
    - [Reward Scale Reference](#reward-scale-reference)
    - [Penalty 1 — Steering Oscillation](#penalty-1--steering-oscillation)
    - [Penalty 2 — Brake Tap](#penalty-2--brake-tap)
    - [Penalty 3 — Neo Slide at Low Speed](#penalty-3--neo-slide-at-low-speed)
24. [Braking Aggression Conditioning](#24-braking-aggression-conditioning)
    - [State Conditioning (UVFA)](#part-1--state-conditioning-uvfa)
    - [Brier-Score Reward Penalty](#part-2--brier-score-reward-penalty)
25. [Risk Tolerance Conditioning](#25-risk-tolerance-conditioning)
    - [State Conditioning](#part-1--state-conditioning)
    - [CVaR Quantile Bias](#part-2--cvar-quantile-bias)
    - [VCP-Distance Brier-Score Reward](#part-3--vcp-distance-brier-score-reward)

---

## Figures & Architecture Diagrams

> **Placeholder — figures to be added here.**

Planned additions:

- [ ] Full system architecture diagram (Service 1 → Service 2 → Linesight pipeline)
- [ ] Driver profile conditioning diagram (braking aggression + risk tolerance inputs)
- [ ] Training reward breakdown plot (base reward vs. style penalties over training)
- [ ] IQN quantile distribution plots from trained runs
- [ ] CVaR quantile range visualization for different risk_tolerance values
- [ ] Lap time comparison: baseline Linesight vs. driver-conditioned RacerHelper

---

## 23. Human-Likeness Penalties

**Files:** `trackmania_rl/buffer_management.py`, `config_files/config.py`, `trackmania_rl/multiprocess/learner_process.py`

These three penalty terms discourage behaviors that are physically possible in the simulator but impossible or unnatural for a human driver. The goal is not to slow the agent down — it is to constrain the **policy space** to solutions a human could realistically execute, which is a prerequisite for using the agent as a driver assistance or coaching tool.

---

### Reward Scale Reference

At ~100 km/h:

| Component | Value per 50 ms step |
|---|---|
| Time penalty | **−0.060** |
| Progress reward (~1.4 m at 100 km/h) | **+0.014** |
| Net reward at racing speed | **≈ −0.032 to −0.046** |

All three penalties use coefficient **−0.05** — roughly 1.0–1.5× the magnitude of one step's net reward. Large enough to shape behavior, small enough not to destroy the primary signal.

---

### Penalty 1 — Steering Oscillation

**Config key:** `humanlike_oscillation_penalty_schedule` (default `[(0, -0.05)]`)

**What it penalizes:** Rapid left↔right direction reversals. Human minimum reaction time for a deliberate direction change is ~150–200 ms, making single-step reversals physiologically impossible.

**Detection:** Prefix-sum sliding window over the rollout. A flip is counted whenever the steering direction reverses (neutral steps ignored). Window size `W = 4` steps = 200 ms.

**Penalty formula:**
```
penalty(i) = humanlike_oscillation_penalty × max(0, flips_in_window − 1)
```

One free flip per 200 ms window is allowed (normal cornering). Every flip beyond the first adds a penalty unit:

| Flips in 200 ms | Multiplier | Penalty | Interpretation |
|---|---|---|---|
| 0 or 1 | 0 | 0 | Normal driving |
| 2 | 1 | −0.05 | L→R→L — borderline |
| 3 | 2 | −0.10 | L→R→L→R — clearly inhuman |
| 4 | 3 | −0.15 | Maximum tapping rate |

---

### Penalty 2 — Brake Tap

**Config key:** `humanlike_brake_tap_penalty_schedule` (default `[(0, -0.05)]`)

**What it penalizes:** Brake presses held for fewer than **3 consecutive steps (150 ms)**. Any brake event shorter than this is a micro-tap that no human could reproduce intentionally.

**Detection:** Forward-scan pre-pass tracks `brake_hold` (running count of consecutive brake steps). When brake turns off with `0 < brake_hold < 3`, the release step is tagged. This is an **event penalty** — one hit per tap occurrence, not per step.

Via n-step return (`n=3`):
```
total effective signal ≈ −0.05 × (1 + γ + γ²) ≈ −0.15
```

The agent strongly associates the entire short-brake sequence with the penalty, not just the release step.

---

### Penalty 3 — Neo Slide at Low Speed

**Config key:** `humanlike_low_speed_slide_penalty_schedule` (default `[(0, -0.05)]`)

**What it penalizes:** Any wheel in a sliding state while forward speed is below **10 m/s (36 km/h)**. At high speed, sliding is a legitimate technique (speedslide). At low speed, it is an AI physics exploit — spin-outs, awkward wall recoveries, or start-of-race exploits.

**Detection:** Per-step check using `state_float[21:25]` (sliding flags) and `state_float[58]` (forward speed):

```python
if any(is_sliding) and abs(speed_forward) < 10.0:
    reward_into[i] += humanlike_low_speed_slide_penalty
```

This is a **per-step penalty** — 10 steps of low-speed sliding = 10 × −0.05 = −0.50, roughly equivalent to losing 8 seconds of race time.

---

### Interaction with Existing Rewards

| Reward / Penalty | What it targets | Interaction |
|---|---|---|
| `engineered_neoslide_reward` | Lateral speed ≥ 2 m/s | No conflict — speed threshold (36 km/h) separates them |
| `engineered_speedslide_reward` | Perfect high-speed slides | Entirely orthogonal to low-speed penalty |
| `humanlike_low_speed_slide_penalty` | Sliding + speed < 36 km/h | Fills the gap left by the above two |
| `humanlike_oscillation_penalty` | Input frequency | No overlap with any existing reward |
| `humanlike_brake_tap_penalty` | Input duration | No overlap with any existing reward |

### Tuning

All three are linear schedules `[(step, coefficient), ...]`.

**Enable immediately (current default):**
```python
humanlike_oscillation_penalty_schedule    = [(0, -0.05)]
humanlike_brake_tap_penalty_schedule      = [(0, -0.05)]
humanlike_low_speed_slide_penalty_schedule = [(0, -0.05)]
```

**Ramp in after initial convergence (recommended from scratch):**
```python
humanlike_oscillation_penalty_schedule    = [(0, 0), (500_000, -0.05)]
humanlike_brake_tap_penalty_schedule      = [(0, 0), (500_000, -0.05)]
humanlike_low_speed_slide_penalty_schedule = [(0, 0), (500_000, -0.05)]
```

---

## 24. Braking Aggression Conditioning

**Files:** `config_files/config.py`, `trackmania_rl/tmi_interaction/game_instance_manager.py`, `trackmania_rl/buffer_management.py`

### Motivation

One of the most driver-distinctive behaviors is braking aggression: aggressive drivers brake late and hard; smooth drivers coast and brake gently. A single trained model needs to produce different driving styles by receiving the target profile as a conditioning input.

---

### Part 1 — State Conditioning (UVFA)

`braking_aggression` is appended to the float feature vector as index **184**, a scalar in `[0, 1]`:

| Value | Interpretation |
|---|---|
| 0.0 | Never brakes — pure coasting driver |
| 0.3 | Brakes sparingly, mostly on fast corners (default) |
| 0.5 | Moderate — brakes at most corners |
| 1.0 | Brakes maximally at every braking opportunity |

This is the **Universal Value Function Approximator (UVFA)** pattern (Schaul et al., 2015). The network learns different Q-value distributions for the same physical state depending on the conditioning variable.

**Normalization:** Mean `0.5`, Std `0.3` (covers realistic driver range with ≈1.7σ from each extreme).

---

### Part 2 — Brier-Score Reward Penalty

The **Brier score** is a proper scoring rule for binary events:

```
L_Brier(p, y) = (y − p)²
```

Expected value is minimized if and only if the predicted probability `p` equals the true event probability — i.e., the agent is incentivized to produce the correct braking frequency.

**Per-step reward formula:**
```
r_brake(i) = coeff × (brake(action_i) − braking_aggression)²
```

| Scenario | Penalty |
|---|---|
| `braking_aggression=1.0`, agent brakes | `coeff × (1−1)² = 0` |
| `braking_aggression=1.0`, agent does not brake | `coeff × (0−1)² = coeff` |
| `braking_aggression=0.5`, either outcome | `coeff × (0.5)² = 0.25 × coeff` |
| `braking_aggression=0.0`, agent brakes | `coeff × (1−0)² = coeff` |

With `coeff = −0.05`, the maximum penalty per step is `−0.05`.

**Why Brier score over cross-entropy?** The Brier score is bounded `[0, 1]` with bounded gradients. Cross-entropy diverges as predicted probability → 0 or 1, which would produce unbounded rewards and destabilize IQN training.

### Configuration

```python
braking_aggression = 0.3
humanlike_braking_aggression_reward_schedule = [(0, -0.05)]
```

To ramp in after convergence:
```python
humanlike_braking_aggression_reward_schedule = [(0, 0), (500_000, -0.05)]
```

To change driver profile mid-run (edit `config_copy.py`):
```python
braking_aggression = 0.7
```

---

## 25. Risk Tolerance Conditioning

**Files:** `config_files/config.py`, `trackmania_rl/tmi_interaction/game_instance_manager.py`, `trackmania_rl/buffer_management.py`

### Motivation

After braking aggression, another driver-style dimension is **risk tolerance**: conservative drivers keep safer margins while aggressive drivers take tighter cuts and operate closer to the limit. Risk is defined operationally as the car's deviation from the VCP reference line.

---

### Part 1 — State Conditioning

`risk_tolerance` is appended to the float feature vector as index **185**, a scalar in `[0, 1]`:

| Value | Interpretation |
|---|---|
| 0.0 | Maximally conservative — stays close to the reference line |
| 0.5 | Neutral / balanced |
| 1.0 | Maximally aggressive — larger deviation from the reference line |

**Normalization:** Same mechanism as `braking_aggression` — part of the float input normalization system in `state_normalization.py`.

---

### Part 2 — CVaR Quantile Bias

Risk tolerance also controls which part of the return distribution is used at inference time. IQN samples quantiles from a risk-conditioned range:

```python
τ ~ U[0.5 × risk_tolerance, 0.5 × risk_tolerance + 0.5]
```

| `risk_tolerance` | Quantile range | Interpretation |
|---|---|---|
| 0.0 | τ ~ U[0.00, 0.50] | Pessimistic / CVaR — conservative decision rule |
| 0.5 | τ ~ U[0.25, 0.75] | Neutral / balanced |
| 1.0 | τ ~ U[0.50, 1.00] | Optimistic / risk-seeking — values upside potential |

A low `risk_tolerance` makes the agent evaluate actions using the lower part of the return distribution (pessimistic). A high value uses the upper part (optimistic). The same trained model changes its decision style without requiring a separate model per profile.

---

### Part 3 — VCP-Distance Brier-Score Reward

Risk is defined as the normalized distance to the current VCP:

```python
risk_proxy = min(||state_float[62:65]|| / risk_tolerance_vcp_dist_max, 1.0)
```

| `risk_proxy` | Interpretation |
|---|---|
| ≈ 0 | Car is close to the reference line |
| ≈ 1 | Car is near `risk_tolerance_vcp_dist_max` metres (default: 15 m) away |

**Reward formula:**
```python
r_risk(i) = coeff × (risk_proxy − risk_tolerance)²
```

The penalty is minimized when `risk_proxy ≈ risk_tolerance`:

| `risk_tolerance` | Minimum penalty occurs at | Interpretation |
|---|---|---|
| 0.0 | `risk_proxy ≈ 0.0` | Stay close to the reference line |
| 0.5 | `risk_proxy ≈ 0.5` | ~7.5 m deviation |
| 1.0 | `risk_proxy ≈ 1.0` | ~15 m deviation |

### Configuration

```python
risk_tolerance = 0.5
humanlike_risk_tolerance_reward_schedule = [(0, -0.05)]
risk_tolerance_vcp_dist_max = 15.0
```

To ramp in after convergence:
```python
humanlike_risk_tolerance_reward_schedule = [(0, 0), (500_000, -0.05)]
```

To change risk profile mid-run (edit `config_copy.py`):
```python
risk_tolerance = 0.8
```

---

### Relationship Between All Style Variables

| Variable | What it controls | Mechanism |
|---|---|---|
| `braking_aggression` | How often the agent brakes | State conditioning + Brier-score penalty on brake frequency |
| `risk_tolerance` | How far from the reference line the agent drives | State conditioning + CVaR quantile bias + VCP-distance Brier penalty |
| Human-likeness penalties | Prevents inhuman simulator exploits | Reward penalties on oscillation, tap-braking, low-speed slides |

The three are fully orthogonal and can be tuned independently. Together they define a driver profile that constrains the agent to a human-achievable racing style.

---

### Total Reward Summary

```python
r_total = r_base                       # time penalty + VCP progress
        + r_engineered                 # optional: speedslide, neoslide, kamikaze
        + r_humanlike_oscillation      # steering oscillation penalty
        + r_humanlike_brake_tap        # brake tap duration penalty
        + r_humanlike_low_speed_slide  # low-speed slide penalty
        + r_brake_aggression           # Brier-score brake frequency penalty
        + r_risk_tolerance             # Brier-score VCP-distance penalty
```
