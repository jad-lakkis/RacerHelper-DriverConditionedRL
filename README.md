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

## Team
- Jad Al Lakkis  
- Ibrahim Khaled
