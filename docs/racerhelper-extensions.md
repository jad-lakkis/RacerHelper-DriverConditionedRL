# RacerHelper — Extensions & Driver Conditioning (Sections 23–29)

> What we added on top of Linesight to produce a **personalized, human-achievable ghost** for a specific driver.  
> Base model: [Linesight](https://github.com/linesight-rl/linesight) — see [linesight-technical.md](./linesight-technical.md) for core architecture.

---

## Table of Contents

23. [Human-Likeness Penalties](#23-human-likeness-penalties)
    - [Reward Scale Reference](#reward-scale-reference)
    - [Penalty 1 — Steering Oscillation](#penalty-1--steering-oscillation)
    - [Penalty 2 — Brake Tap](#penalty-2--brake-tap)
    - [Penalty 3 — Steering Tap](#penalty-3--steering-tap)
    - [Penalty 4 — Accelerator Tap](#penalty-4--accelerator-tap)
    - [Penalty 5 — Neo Slide at Low Speed](#penalty-5--neo-slide-at-low-speed)
    - [Interaction with Existing Rewards](#interaction-with-existing-rewards)
    - [Tuning](#tuning)
24. [Braking Aggression Conditioning](#24-braking-aggression-conditioning)
25. [Risk Tolerance Conditioning](#25-risk-tolerance-conditioning)
    - [Part 1 — CVaR Quantile Bias](#part-1--cvar-quantile-bias)
    - [Part 2 — VCP-Distance Brier-Score Reward](#part-2--vcp-distance-brier-score-reward)
26. [Oversteer / Understeer Conditioning](#26-oversteer--understeer-conditioning)
27. [Corner Entry Speed Conditioning](#27-corner-entry-speed-conditioning)
28. [Extract Driver Profile Script](#28-extract-driver-profile-script)
29. [Total Reward Summary](#29-total-reward-summary)

---

## 23. Human-Likeness Penalties

**Files:** `trackmania_rl/buffer_management.py`, `config_files/config.py`

Five penalty terms discourage behaviors that are physically possible in the simulator but impossible or unnatural for a human driver. The goal is not to slow the agent down — it is to constrain the **policy space** to solutions a human could realistically execute, which is a prerequisite for using the agent as a coaching tool.

---

### Reward Scale Reference

At ~100 km/h:

| Component | Value per 50 ms step |
|---|---|
| Time penalty | **−0.060** |
| Progress reward (~1.4 m at 100 km/h) | **+0.014** |
| Net reward at racing speed | **≈ −0.032 to −0.046** |

All five penalties use coefficient **−0.05** — roughly 1–1.5× the magnitude of one step's net reward. Large enough to shape behavior, small enough not to destroy the primary signal.

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

### Penalty 3 — Steering Tap

**Config key:** `humanlike_steer_tap_penalty_schedule` (default `[(0, -0.05)]`)

**What it penalizes:** Left or right presses held fewer than **3 consecutive steps (150 ms)**. Mirrors the brake tap logic applied to steering inputs.

**Detection:** Left and right holds are tracked independently so a direct L→R transition (no neutral frame) correctly tags the left release at the step right begins. This penalty is orthogonal to the oscillation penalty: oscillation catches rapid direction alternation by frequency; tap catches any individual press that is too short regardless of direction.

---

### Penalty 4 — Accelerator Tap

**Config key:** `humanlike_accel_tap_penalty_schedule` (default `[(0, -0.05)]`)

**What it penalizes:** Throttle presses held fewer than **3 consecutive steps (150 ms)**. No human driver blips the gas that briefly in a racing context.

**Detection:** Same forward-scan pattern as the brake tap pre-pass.

---

### Penalty 5 — Neo Slide at Low Speed

**Config key:** `humanlike_low_speed_slide_penalty_schedule` (default `[(0, -0.05)]`)

**What it penalizes:** Any wheel in a sliding state while forward speed is below **10 m/s (36 km/h)**. At high speed, sliding is a legitimate technique (speedslide). At low speed, it is an AI physics exploit — spin-outs, awkward wall recoveries, or start-of-race exploits that no human driver produces intentionally.

**Detection:** Per-step check using `state_float[21:25]` (sliding flags) and `state_float[58]` (forward speed):

```python
if any(is_sliding) and abs(speed_forward) < 10.0:
    reward_into[i] += humanlike_low_speed_slide_penalty
```

This is a **per-step penalty** — 10 steps of low-speed sliding = 10 × −0.05 = −0.50, roughly equivalent to about 8 time-penalty steps (~0.4 seconds of race time).

---

### Interaction with Existing Rewards

| Reward / Penalty | What it targets | Interaction |
|---|---|---|
| `engineered_neoslide_reward` | Lateral speed ≥ 2 m/s AND forward speed ≥ 10 m/s | No conflict — the forward-speed guard was added specifically to prevent overlap with the low-speed penalty |
| `engineered_speedslide_reward` | Perfect high-speed slides | Entirely orthogonal to low-speed penalty |
| `humanlike_low_speed_slide_penalty` | Sliding + speed < 36 km/h | Fills the gap left by the above two |
| `humanlike_oscillation_penalty` | Input frequency | No overlap with any existing reward |
| `humanlike_brake_tap_penalty` | Input duration | No overlap with any existing reward |
| `humanlike_steer_tap_penalty` | Input duration | Orthogonal to oscillation penalty (duration vs. frequency) |
| `humanlike_accel_tap_penalty` | Input duration | No overlap with any existing reward |

### Tuning

All five are linear schedules `[(step, coefficient), ...]`.

**Enable immediately (current default):**
```python
humanlike_oscillation_penalty_schedule     = [(0, -0.05)]
humanlike_brake_tap_penalty_schedule       = [(0, -0.05)]
humanlike_steer_tap_penalty_schedule       = [(0, -0.05)]
humanlike_accel_tap_penalty_schedule       = [(0, -0.05)]
humanlike_low_speed_slide_penalty_schedule = [(0, -0.05)]
```

**Ramp in after initial convergence (recommended when training from scratch):**
```python
humanlike_oscillation_penalty_schedule     = [(0, 0), (500_000, -0.05)]
humanlike_brake_tap_penalty_schedule       = [(0, 0), (500_000, -0.05)]
humanlike_steer_tap_penalty_schedule       = [(0, 0), (500_000, -0.05)]
humanlike_accel_tap_penalty_schedule       = [(0, 0), (500_000, -0.05)]
humanlike_low_speed_slide_penalty_schedule = [(0, 0), (500_000, -0.05)]
```

---

## 24. Braking Aggression Conditioning

**Files:** `config_files/config.py`, `trackmania_rl/buffer_management.py`

### Motivation

One of the most driver-distinctive behaviors is braking aggression: aggressive drivers brake late and hard; smooth drivers coast and brake gently. The system shapes the agent's braking frequency toward a target value `braking_aggression ∈ [0, 1]` via a Brier-score reward penalty.

### Brier-Score Reward Penalty

The **Brier score** is the unique proper scoring rule for binary events:

```
L_Brier(p, y) = (y − p)²
```

Its expectation over the policy is minimized if and only if the agent's empirical brake frequency equals `braking_aggression`. This makes it the correct loss for driving the distribution of brake actions toward a target frequency.

**Per-step reward formula:**
```
r_brake(i) = coeff × (brake(action_i) − braking_aggression)²
```

| `braking_aggression` | Agent action | Penalty |
|---|---|---|
| 1.0 | brakes | `coeff × (1−1)² = 0` |
| 1.0 | does not brake | `coeff × (0−1)² = coeff` |
| 0.5 | either | `coeff × (0.5)² = 0.25 × coeff` |
| 0.0 | brakes | `coeff × (1−0)² = coeff` |

With `coeff = −0.05`, the maximum penalty per step is `−0.05`.

**Why Brier score over cross-entropy?** The Brier score is bounded `[0, 1]` with bounded gradients. Cross-entropy diverges as the predicted probability → 0 or 1, which would produce unbounded rewards and destabilize IQN training.

### Configuration

```python
braking_aggression = 0.8  # [0, 1]; use extract_driver_profile.py to derive from a replay
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

**Files:** `config_files/config.py`, `trackmania_rl/agents/iqn.py`, `trackmania_rl/buffer_management.py`

### Motivation

Risk tolerance captures how aggressively a driver cuts corners and how far they deviate from the centerline. Conservative drivers follow the reference line closely; aggressive drivers take tight apex cuts. The variable `risk_tolerance ∈ [0, 1]` conditions the agent through two complementary mechanisms.

---

### Part 1 — CVaR Quantile Bias

At inference time, IQN samples quantiles from a risk-conditioned range:

```python
τ ~ U[0.5 × risk_tolerance, 0.5 × risk_tolerance + 0.5]
```

| `risk_tolerance` | Quantile range | Interpretation |
|---|---|---|
| 0.0 | τ ~ U[0.00, 0.50] | Pessimistic / CVaR — conservative decision rule |
| 0.5 | τ ~ U[0.25, 0.75] | Neutral / balanced |
| 1.0 | τ ~ U[0.50, 1.00] | Optimistic / risk-seeking — values upside potential |

This is implemented in `trackmania_rl/agents/iqn.py → Inferer.infer_network()`. The same trained model changes its decision style at inference without retraining.

---

### Part 2 — VCP-Distance Brier-Score Reward

Risk is operationalized as the car's normalized 3D distance from the current VCP:

```python
risk_proxy = min(||state_float[62:65]|| / risk_tolerance_vcp_dist_max, 1.0)
```

| `risk_proxy` | Interpretation |
|---|---|
| ≈ 0 | Car is on the reference line |
| ≈ 1 | Car is `risk_tolerance_vcp_dist_max` metres away (default: 15 m) |

**Reward formula:**
```python
r_risk(i) = coeff × (risk_proxy − risk_tolerance)²
```

The Brier score drives `risk_proxy → risk_tolerance` at equilibrium:

| `risk_tolerance` | Minimum penalty at | Interpretation |
|---|---|---|
| 0.0 | `risk_proxy ≈ 0.0` | Stay close to the reference line |
| 0.5 | `risk_proxy ≈ 0.5` | ~7.5 m lateral deviation |
| 1.0 | `risk_proxy ≈ 1.0` | ~15 m lateral deviation |

### Configuration

```python
risk_tolerance = 0.2  # [0, 1]; 0 = conservative, 1 = aggressive
humanlike_risk_tolerance_reward_schedule = [(0, -0.05)]
risk_tolerance_vcp_dist_max = 15.0  # metres; normalisation denominator for VCP distance
```

To change risk profile mid-run:
```python
risk_tolerance = 0.8
```

---

## 26. Oversteer / Understeer Conditioning

**Files:** `config_files/config.py`, `trackmania_rl/buffer_management.py`

### Motivation

Cornering style — the degree to which a driver induces oversteer (rear steps out) vs. understeer (front washes out) — is a distinctive driver characteristic. The variable `oversteer_understeer_score ∈ [−5, 5]` conditions the agent via a per-step alignment reward. Setting it to `0` disables this term entirely.

### Signal Construction

Two mutually exclusive per-step signals are derived from the car's velocity state:

| Signal | Condition | Value |
|---|---|---|
| **Oversteer** | `\|v_lat\| / \|v_fwd\|` is high | `clip(slip / 0.3, 0, 1) → [0, 1]` |
| **Understeer** | steering pressed AND slip ratio `< 0.04` | `1.0` (binary flag) |

Combined into a single signed signal:
```
o_signal = clip(oversteer_component − understeer_component, −1, 1)
```

- `o = +1`: full oversteer (lateral slip saturated)
- `o = −1`: full understeer (steering applied, near-zero lateral response)
- `o =  0`: neutral tracking

Suppressed below 10 m/s to avoid low-speed noise.

### Reward

```
r = coeff × (score / 5) × o_signal
```

The reward is **positive when the agent's cornering style aligns with the score**:

| Score | Behavior | Reward |
|---|---|---|
| `s > 0` (oversteer driver) | agent oversteers (`o > 0`) | positive |
| `s < 0` (understeer driver) | agent understeers (`o < 0`) | positive |
| any | misaligned style | negative |
| `s = 0` | disabled | 0 |

Note: the config coefficient is **positive** (`+0.05`). The sign of the reward follows from `(score/5) × o_signal`.

### Configuration

```python
oversteer_understeer_score = 0.0  # [-5, 5]; 0 = disabled
humanlike_oversteer_understeer_reward_schedule = [(0, 0.05)]
```

Slip thresholds in `trackmania_rl/buffer_management.py`:
```python
_OVERSTEER_SLIP_SAT  = 0.3   # |v_lat|/|v_fwd| at which oversteer signal saturates
_UNDERSTEER_SLIP_MAX = 0.04  # |v_lat|/|v_fwd| below which understeer fires when steering
```

---

## 27. Corner Entry Speed Conditioning

**Files:** `config_files/config.py`, `trackmania_rl/buffer_management.py`

### Motivation

How much speed a driver carries into a corner is a key style dimension. Late brakers enter fast (`ratio ≈ 0.84`); conservative drivers scrub speed early (`ratio ≈ 0.3`). The variable `corner_entry_speed_ratio ∈ [0, 1]` = median corner-entry speed / peak lap speed shapes agent behavior via a sparse Brier-score penalty.

### Corner Detection

A corner entry is detected when the curvature proxy κ crosses a rising edge above `CORNER_CURV_THRESH = 0.010 1/m`:

```
κ = |angular_velocity_y| / max(|v_fwd|, 1.0)   (state_float indices 54, 58)
```

A minimum gap of 10 steps (500 ms) is enforced between detections. This mirrors the detection logic in `scripts/extract_driver_profile.py` so the profiler and the reward function measure the same corners.

### Reward

At each detected corner entry, a Brier-score penalty fires:

```
r = coeff × (entry_ratio − corner_entry_speed_ratio)²
```

where `entry_ratio = speed_at_entry / rollout_peak_speed`.

With `coeff = −0.2`, the worst-case per-entry cost is `−0.2` — comparable to ~3 steps of the time penalty. There are typically 10–30 corner entries per lap, so the total influence is meaningful but not dominant.

### Configuration

```python
corner_entry_speed_ratio = 0.84  # [0, 1]; use extract_driver_profile.py to derive from a replay
humanlike_corner_entry_speed_reward_schedule = [(0, -0.2)]
```

---

## 28. Extract Driver Profile Script

**File:** `linesight/scripts/extract_driver_profile.py`

Converts a Trackmania `.Replay.Gbx` into the driver-style values used by the conditioning system.

```bash
cd linesight/
PYTHONPATH=. python scripts/extract_driver_profile.py <replay.Replay.Gbx>
```

### Outputs

| Output | Config key | Range | Description |
|---|---|---|---|
| `braking_aggression` | `braking_aggression` | `[0, 1]` | Context-aware brake frequency (conditioned on corner-approach context) |
| `oversteer_understeer_score` | `oversteer_understeer_score` | `[-5, 5]` | Mean o_signal × 5 across the lap |
| `corner_entry_speed_level` | — | `low / medium / high` | Human-readable label (`high` > 0.60, `medium` 0.35–0.60, `low` < 0.35) |
| `corner_entry_speed_ratio` | `corner_entry_speed_ratio` | `[0, 1]` | Numeric ratio to paste into config |

The script prints a ready-to-paste config block:

```
═══════════════════════════════════════════════════════
  DRIVER PROFILE
═══════════════════════════════════════════════════════
  braking_aggression         = 0.810
  oversteer_understeer_score = -0.60
  corner_entry_speed_level   = high
  corner_entry_speed_ratio   = 0.840

  ── paste into config_files/config.py ──
  braking_aggression         = 0.81
  oversteer_understeer_score = -0.6
  corner_entry_speed_ratio   = 0.84
```

### How Braking Aggression is Measured

Braking frequency is conditioned on corner-approach context: the denominator counts only samples where the car is above 50 km/h **and** a corner exists within the next 200 m ahead. This avoids penalising a driver for not braking on long straights. Falls back to a speed²-weighted brake fraction on pure-speed tracks with no detectable corners.

### How Oversteer / Understeer is Measured

A Gaussian-smoothed velocity vector (σ = 6 steps / 600 ms) is used as the forward reference, so the perpendicular component of the raw velocity captures genuine lateral slip without circular-reference bias. The signal matches the constants in `buffer_management.py` (`SLIP_SAT = 0.3`, `UNDERSTEER_MAX = 0.04`).

---

## 29. Total Reward Summary

All reward components active during a training step:

```python
r_total = r_base                        # time penalty + VCP progress
        + r_engineered                  # optional: speedslide, neoslide, kamikaze, close_to_vcp
        + r_humanlike_oscillation       # steering oscillation penalty
        + r_humanlike_brake_tap         # brake tap duration penalty
        + r_humanlike_steer_tap         # steering tap duration penalty
        + r_humanlike_accel_tap         # accelerator tap duration penalty
        + r_humanlike_low_speed_slide   # low-speed slide penalty
        + r_brake_aggression            # Brier-score brake frequency penalty
        + r_risk_tolerance              # Brier-score VCP-distance penalty
        + r_oversteer_understeer        # cornering style alignment reward
        + r_corner_entry_speed          # Brier-score corner entry speed penalty
```

CVaR quantile bias is applied at **inference time** (not a reward): `τ ~ U[0.5×risk_tolerance, 0.5×risk_tolerance + 0.5]`.

### Driver Profile Variable Summary

| Variable | Range | Mechanism | Config key |
|---|---|---|---|
| `braking_aggression` | `[0, 1]` | Brier-score penalty on brake frequency | `humanlike_braking_aggression_reward_schedule` |
| `risk_tolerance` | `[0, 1]` | CVaR quantile bias at inference + Brier-score VCP-distance penalty | `humanlike_risk_tolerance_reward_schedule` |
| `oversteer_understeer_score` | `[-5, 5]` | Per-step alignment reward on slip signal | `humanlike_oversteer_understeer_reward_schedule` |
| `corner_entry_speed_ratio` | `[0, 1]` | Brier-score penalty on corner entry speed | `humanlike_corner_entry_speed_reward_schedule` |

The four style variables are fully orthogonal and can be tuned independently. The five human-likeness penalties constrain the agent to a human-achievable action space. Together they define a driver profile that is extracted from a replay via `extract_driver_profile.py` and pasted directly into `config_files/config.py`.
