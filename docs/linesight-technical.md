# Linesight — Technical Reference (Sections 1–22)

> The original Linesight base system — a deep RL agent that learns to drive Trackmania Nations Forever at superhuman speeds.  
> Algorithm: **IQN** · Architecture: Dueling + DDQN · Framework: PyTorch  
> Source: [linesight-rl/linesight](https://github.com/linesight-rl/linesight)

For RacerHelper-specific extensions (driver conditioning, human-likeness penalties), see [racerhelper-extensions.md](./racerhelper-extensions.md).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Workflow Map](#3-system-workflow-map)
4. [Multiprocessing Architecture](#4-multiprocessing-architecture)
5. [Configuration System](#5-configuration-system)
6. [Game Interaction Layer](#6-game-interaction-layer)
7. [State Space — What the Agent Sees](#7-state-space--what-the-agent-sees)
8. [Action Space — What the Agent Can Do](#8-action-space--what-the-agent-can-do)
9. [Reward Function](#9-reward-function)
10. [Map & Virtual Checkpoint System](#10-map--virtual-checkpoint-system)
11. [Neural Network Architecture](#11-neural-network-architecture)
12. [IQN Algorithm — Theory & Implementation](#12-iqn-algorithm--theory--implementation)
13. [Replay Buffer & Mini-Race Logic](#13-replay-buffer--mini-race-logic)
14. [Training Loop — Step by Step](#14-training-loop--step-by-step)
15. [Exploration Strategy](#15-exploration-strategy)
16. [Target Network & Soft Updates](#16-target-network--soft-updates)
17. [Hyperparameter Reference](#17-hyperparameter-reference)
18. [Scheduling System](#18-scheduling-system)
19. [Checkpointing & Saving](#19-checkpointing--saving)
20. [TensorBoard Metrics](#20-tensorboard-metrics)
21. [Utilities & Helper Modules](#21-utilities--helper-modules)
22. [File-by-File Reference](#22-file-by-file-reference)

---

## 1. Project Overview

Linesight trains an RL agent to race in **Trackmania Nations Forever** as fast as possible. The agent receives a grayscale screenshot of the game and a hand-crafted float feature vector at each decision point, and outputs a discrete driving action (gas, brake, steer, combinations). The goal is to minimize race time.

Key design choices:
- **IQN** replaces a standard DQN value head with a full return-distribution head, giving the agent richer uncertainty awareness and enabling risk-sensitive policies.
- A **"mini-race" / clipped-horizon** trick lets the agent optimize a bounded-time Q-value (7 seconds) with `γ = 1`, avoiding the need to discount very long-horizon rewards.
- **Virtual checkpoints (VCPs)** along the track centerline replace sparse real checkpoint rewards with a dense per-meter progress signal.
- **Parallel rollout workers** interact with multiple game instances simultaneously while a single GPU learner trains on the collected data.
- The configuration file can be hot-edited during training and changes take effect without restarting — the running `config_copy.py` is reloaded periodically.

---

## 2. Repository Structure

```
linesight/
├── scripts/
│   ├── train.py                         # Entry point — spawns all processes
│   └── tools/                           # Offline utilities (GBX parsing, video generation)
│
├── config_files/
│   ├── config.py                        # Master config (user-editable)
│   ├── config_copy.py                   # Live config (auto-copied, hot-reloaded)
│   ├── inputs_list.py                   # Action definitions (12 actions)
│   ├── state_normalization.py           # Float input mean/std for normalization
│   └── user_config.py                   # Machine-specific paths & settings
│
├── trackmania_rl/
│   ├── agents/iqn.py                    # IQN_Network, Trainer, Inferer classes
│   ├── multiprocess/
│   │   ├── learner_process.py           # GPU training loop + TensorBoard
│   │   └── collector_process.py         # Game-playing worker processes
│   ├── tmi_interaction/
│   │   ├── game_instance_manager.py     # TMInterface bridge — rollout() function
│   │   ├── tminterface2.py              # Low-level TMInterface socket protocol
│   │   └── Python_Link.as               # AngelScript plugin (injected into the game)
│   ├── buffer_management.py             # Converts rollouts to Experience objects
│   ├── buffer_utilities.py              # Buffer construction, collate fn, mini-race logic
│   ├── reward_shaping.py                # Speedslide quality reward helper
│   ├── map_loader.py                    # VCP loading, GBX parsing, CP sync
│   └── utilities.py                     # NN utils, schedules, checkpointing
│
├── maps/                                # Pre-computed VCP centerline arrays (.npy)
├── save/                                # Auto-created; stores weights + stats
└── tensorboard/                         # TensorBoard event files
```

---

## 3. System Workflow Map

```
scripts/train.py
│
│  1. Copies config.py → config_copy.py
│  2. Creates shared memory objects (shared_network, locks, queues)
│  3. Spawns N collector processes
│  4. Runs learner_process_fn() in main process
│
├──────────────────────────────────────────────────────────────────────┐
│  COLLECTOR PROCESS (×N)                                              │
│                                                                      │
│   ┌─ game_instance_manager.py                                        │
│   │  rollout():                                                       │
│   │  ├─ tminterface2.py  (socket: 50 ms tick)                        │
│   │  ├─ Grabs grayscale frame (160×120)                              │
│   │  ├─ Reads sim state (position, velocity, orientation)            │
│   │  ├─ Constructs float feature vector (188,)                       │
│   │  ├─ agents/iqn.py  Inferer → IQN_Network.forward()              │
│   │  └─ Sends action back to game                                    │
│   │                                                                   │
│   └─ rollout_queue.put(rollout_results)                               │
└──────────────────────────────────────────────────────────────────────┘
         │ (multiprocessing.Queue)
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LEARNER PROCESS (main process)                                      │
│                                                                      │
│   1. Receive rollout from collector queue                            │
│   2. buffer_management → fill_buffer_from_rollout_with_n_steps_rule │
│   3. buffer_utilities  → buffer_collate_function (mini-race logic)  │
│   4. agents/iqn.py Trainer.train_on_batch()                          │
│      ├─ online_network.forward(state) → Q(s,a,τ3)                   │
│      ├─ target_network.forward(next)  → Q(s',a',τ2)                 │
│      ├─ iqn_loss() — pinball Huber loss                             │
│      └─ RAdam optimizer step                                         │
│   5. Every 10 batches: push weights to shared_network               │
│   6. Every 2048 memories: soft update target_network                │
│   7. Every 5 min: TensorBoard scalars + save checkpoint             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Multiprocessing Architecture

| Object | Type | Purpose |
|---|---|---|
| `uncompiled_shared_network` | `IQN_Network` in shared memory | Pass updated weights from learner → collectors |
| `shared_network_lock` | `Lock` | Prevent race condition on network read/write |
| `game_spawning_lock` | `Lock` | Serialize game launches |
| `shared_steps` | `mp.Value(c_int64)` | Atomic frame counter for epsilon/LR schedules |
| `rollout_queues` | `[mp.Queue]` | One queue per collector |

`config.gpu_collectors_count` (default: `2`) controls how many parallel game instances run. The learner runs in the main process to avoid creating an extra CUDA context.

**Weight sharing:** Learner trains `online_network` → every 10 batches copies to `uncompiled_shared_network` (under lock) → each collector pulls every 20 actions into its `inference_network`.

---

## 5. Configuration System

Two config files coexist:

- `config.py` — user-facing master config tracked in git. Changes during a run have **no effect**.
- `config_copy.py` — live copy created at startup. Both learner and collectors call `importlib.reload(config_copy)` on every iteration → **editing this file applies changes on the fly** without restarting or clearing the replay buffer.

| Section | Key Variables |
|---|---|
| Image input | `W_downsized=160`, `H_downsized=120` |
| Timing | `tm_engine_step_per_action=5` (50ms/action) |
| Float feature dim | `float_input_dim = 188` |
| Network sizes | `float_hidden_dim=256`, `dense_hidden_dimension=1024`, `iqn_embedding_dimension=64` |
| IQN | `iqn_n=8` (training quantiles), `iqn_k=32` (inference quantiles), `iqn_kappa=5e-3` |
| Replay | `number_times_single_memory_is_used_before_discard=32` |
| Optimization | `batch_size=512`, RAdam with `adam_beta1=0.9`, `adam_beta2=0.999` |
| Mini-race | `temporal_mini_race_duration_ms=7000` → 140 steps |

---

## 6. Game Interaction Layer

Linesight hooks into Trackmania Nations Forever via **TMInterface 2**, a modding framework that exposes the game's simulation state over a local socket. The AngelScript plugin `Python_Link.as` is injected at startup.

### `GameInstanceManager.rollout()`

1. **Launches / reconnects** to the game if needed.
2. **Loads the map** via `iface.execute_command("map <path>")`.
3. **Rewinds to a saved start state** for fast episode restarts.
4. **Event loop** every 50ms:
   - `SC_RUN_STEP_SYNC` — game paused, state read, action computed and applied.
   - `SC_REQUESTED_FRAME_SYNC` — `160×120` BGRA frame grabbed, converted to grayscale, passed to policy.
   - `SC_CHECKPOINT_COUNT_CHANGED_SYNC` — race finish detected (deliberately prevented to record stats cleanly).
5. **Termination:** race finished, 300 s elapsed, or no VCP progress for 2 s.
6. Returns `rollout_results` dict + `end_race_stats`.

---

## 7. State Space — What the Agent Sees

### Image input — `(1, 120, 160)` uint8

Grayscale screenshot at 160×120. Normalized to `[-1, 1]` via `(img.float() - 128) / 128`. Game UI disabled during rollouts.

### Float feature vector — `(188,)` float32

All features in the car's local reference frame:

| Index | Feature | Description |
|---|---|---|
| 0 | Mini-race timer | Steps elapsed in the 7-second window. Injected at collation time. |
| 1–20 | Previous 5 actions | 4 binary flags × 5 steps: accelerate, brake, left, right |
| 21–36 | Wheel & gear state | Per-wheel: sliding flag, ground contact, damper absorb; gearbox state, gear, RPM |
| 37–52 | Contact material types | 4 physics behavior types × 4 wheels (one-hot) |
| 53–55 | Angular velocity | Car angular velocity in car frame (3D) |
| 56–58 | Linear velocity | Car velocity in car frame; index 58 = forward speed |
| 59–61 | Map up-vector | World Y axis in car frame |
| 62–181 | VCP coordinates | 40 upcoming VCP positions relative to car (~200 m look-ahead) |
| 182 | Distance to finish | Meters remaining, capped at 700 m |
| 183 | `is_freewheeling` | Boolean: engine disconnected from wheels |

---

## 8. Action Space — What the Agent Can Do

12 discrete actions combining `{accelerate, brake, left, right}` binary flags. Actions are applied for exactly one step (50 ms).

| Index | Name | Accel | Brake | Left | Right |
|---|---|---|---|---|---|
| 0 | Forward | ✓ | | | |
| 1 | Forward Left | ✓ | | ✓ | |
| 2 | Forward Right | ✓ | | | ✓ |
| 3 | Nothing | | | | |
| 4 | Coast Left | | | ✓ | |
| 5 | Coast Right | | | | ✓ |
| 6 | Brake | | ✓ | | |
| 7 | Brake Left | | ✓ | ✓ | |
| 8 | Brake Right | | ✓ | | ✓ |
| 9 | Brake + Accel | ✓ | ✓ | | |
| 10 | Brake + Accel Left | ✓ | ✓ | ✓ | |
| 11 | Brake + Accel Right | ✓ | ✓ | | ✓ |

---

## 9. Reward Function

Core reward per step:

```
reward[i] = constant_reward_per_ms × ms_per_action
           + reward_per_m_advanced_along_centerline × (meters[i] - meters[i-1])
```

Defaults: `constant_reward_per_ms = -6/5000` (time penalty), `reward_per_m_advanced_along_centerline = 5/500`. These are calibrated so advancing 1 m exactly compensates for the time cost at the slowest viable speed.

Optional engineered rewards (all default to 0, annealed via linear schedules):

| Variable | Trigger | Purpose |
|---|---|---|
| `engineered_speedslide_reward` | All 4 wheels grounded | Reward perfect speedslide quality |
| `engineered_neoslide_reward` | Lateral speed ≥ 2 m/s | Reward any lateral sliding |
| `engineered_kamikaze_reward` | ≤1 wheel grounded | Reward risky maneuvers |
| `engineered_close_to_vcp_reward` | Always | Soft penalty for distance to current VCP |

**Potential-based shaping** (Ng et al., 1999):

```
adjusted_reward = raw_reward + γ × Φ(next_state) − Φ(state)
```

where `Φ(state) = shaped_reward_dist_to_cur_vcp × clip(||state[62:65]||, min, max)`.

---

## 10. Map & Virtual Checkpoint System

VCPs are 3D points along the track centerline, spaced 0.5 m apart, stored as `.npy` files. Generated offline from a ghost replay via `scripts/tools/gbx_to_vcp.py`:

1. Parse ghost position records with `pygbx`.
2. Interpolate to 0.5 m spacing using `make_interp_spline`.
3. Save as `(N, 3)` array.

`load_next_map_zone_centers()` prepends 20 artificial zones before the start and appends 1000 past the finish. The trajectory is then smoothed.

`update_current_zone_idx()` (Numba-JIT) advances the VCP index as the car moves. A real-checkpoint sync prevents the agent from finding shortcuts that skip checkpoints.

---

## 11. Neural Network Architecture

```
Image (1×120×160, float16)
   ├── Conv2D(1→16, 4×4, stride=2)   LeakyReLU
   ├── Conv2D(16→32, 4×4, stride=2)  LeakyReLU
   ├── Conv2D(32→64, 3×3, stride=2)  LeakyReLU
   ├── Conv2D(64→32, 3×3, stride=1)  LeakyReLU
   └── Flatten  →  conv_output (5632,)

Float features (188, float32)
   ├── Linear(188→256)  LeakyReLU
   └── Linear(256→256)  LeakyReLU  →  float_output (256,)

concat = cat(conv_output, float_output)  →  (5888,)

Quantile τ ~ U[0,1]
   ├── cos(π × i × τ) for i=1..64   →  (batch×n_quantiles, 64)
   ├── Linear(64→5888)
   └── LeakyReLU  →  quantile_embedding (5888,)

combined = concat × quantile_embedding  [Hadamard product]

Advantage head A:  Linear(5888→512) LeakyReLU → Linear(512→12)
Value head V:      Linear(5888→512) LeakyReLU → Linear(512→1)

Q(s, a, τ) = V(s, τ) + A(s, a, τ) − mean_a[A(s, a, τ)]
```

All layers use **orthogonal initialization** with LeakyReLU gain. On Linux: `torch.compile(max-autotune)`. On Windows: `torch.jit.script()`.

---

## 12. IQN Algorithm — Theory & Implementation

IQN (Dabney et al. 2018) learns the **full return distribution** Z(s,a) rather than just E[Z] = Q(s,a). For any quantile `τ ∈ (0,1)` the network outputs the τ-th quantile of the return distribution.

### Quantile Huber (Pinball) Loss

```
TD_error[i,j] = target_τ2[i] − output_τ3[j]
loss[i,j]     = Huber_κ(TD_error[i,j]) × |τ3[j] − 𝟙[TD_error<0]|
```

where `Huber_κ(x) = x²/(2κ) if |x|<κ else |x| - κ/2`, `κ = 5e-3`.

### DDQN variant
When `use_ddqn = True`: the online network selects the best next action, the target network evaluates it. Decouples action selection from evaluation to reduce overestimation bias.

### Self-loss normalization
Normalizes the loss by the "self-loss" (IQN loss when target = output = spread of the distribution). Prevents transitions with very broad return distributions from dominating the gradient.

---

## 13. Replay Buffer & Mini-Race Logic

### Experience dataclass
Each stored transition contains: `state_img`, `state_float`, `state_potential`, `action`, `n_steps`, `rewards`, `next_state_img`, `next_state_float`, `next_state_potential`, `gammas`, `terminal_actions`.

### N-step returns
Default `n_steps=3`:
```
reward[i] = Σ_{j=0}^{n-1} γ^j × r[i+j+1]
```

### Mini-race / clipped horizon
Q-values are defined over a **7-second window** with `γ = 1`. At collation time:

1. Random mini-race start time `t ∈ [0, 140]` drawn.
2. `state_float[:, 0]` overwritten with `t`.
3. `next_state_float[:, 0]` overwritten with `t + n_steps`.
4. If `t + n_steps ≥ 140`: treated as terminal (gamma = 0).

This lets the agent use `γ = 1` while avoiding instability from infinite-horizon returns.

### Buffer sizing (staircase schedule)

| Frames | Buffer size |
|---|---|
| 0 | 50,000 |
| 5M | 100,000 |
| 7M | 200,000 |

---

## 14. Training Loop — Step by Step

```
while True:
    1. Receive rollout from any collector queue
    2. Hot-reload config_copy
    3. Compute schedule values (LR, gamma, epsilon, weight_decay)
    4. Update optimizer hyperparameters in-place
    5. Write per-race TensorBoard scalars
    6. Save best run if new record
    7. Convert rollout → Experience objects → add to replay buffer
    8. TRAINING LOOP:
       while buffer sufficient AND training budget remaining:
          - Sample batch → buffer_collate_function
          - Trainer.train_on_batch():
            * online_network(state)  → Q(s,a,τ3)
            * target_network(next)   → Q(s',a',τ2)
            * iqn_loss + self-loss normalization
            * scaler.backward() + clip_grad_norm(30)
            * RAdam step
          - custom_weight_decay
          - Every 10 batches: push weights to shared_network
          - Every 2048 memories: soft_copy_param(target ← online, τ=0.02)
    9. Every 5 min: TensorBoard aggregated stats + save checkpoint
```

---

## 15. Exploration Strategy

1. **Epsilon-greedy** — uniform random action with prob `ε` (decays: 1.0 → 0.1 → 0.03).
2. **Boltzmann** — Gaussian noise added to Q-values: `argmax(Q(s) + 0.01 × randn(12))`.
3. **Greedy** — `argmax(mean_τ[Q(s,a,τ)])`.

Evaluation runs (1 in 5) are purely greedy with no exploration noise.

---

## 16. Target Network & Soft Updates

Every 2048 memories trained on:
```python
target_network ← (1 - 0.02) × target_network + 0.02 × online_network
```

**Periodic network reset** (disabled by default): gently blends online network toward a fresh untrained network to mitigate primacy bias (Nikishin et al., 2022).

---

## 17. Hyperparameter Reference

| Hyperparameter | Value | Description |
|---|---|---|
| `iqn_n` | 8 | Quantiles sampled during training |
| `iqn_k` | 32 | Quantiles sampled during inference |
| `iqn_kappa` | 5e-3 | Huber loss threshold |
| `batch_size` | 512 | Transitions per training batch |
| `n_steps` | 3 | Max n-step return lookahead |
| `gamma_schedule` | 0.999→1.0 | Discount factor (annealed) |
| `soft_update_tau` | 0.02 | Polyak averaging coefficient |
| `clip_grad_norm` | 30 | Max gradient L2 norm |
| `lr_schedule` | 1e-3 → 5e-5 → 1e-5 | Learning rate schedule |
| `weight_decay_lr_ratio` | 1/50 | Weight decay = lr / 50 |
| `temporal_mini_race_duration_ms` | 7000 | Mini-race horizon (7 s = 140 steps) |
| `running_speed` | 80× | Game simulation speed multiplier |

---

## 18. Scheduling System

Three schedule types keyed on `cumul_number_frames_played`:

- **Exponential** (`from_exponential_schedule`) — LR, epsilon, epsilon_boltzmann.
- **Linear** (`from_linear_schedule`) — gamma, engineered reward weights.
- **Staircase** (`from_staircase_schedule`) — memory size, TensorBoard suffix.

---

## 19. Checkpointing & Saving

Every 5 minutes:
```
save/<run_name>/
├── weights1.torch         ← online_network.state_dict()
├── weights2.torch         ← target_network.state_dict()
├── optimizer1.torch       ← RAdam state
├── scaler.torch           ← AMP GradScaler state
└── accumulated_stats.joblib
```

On a new all-time best race time:
```
save/<run_name>/best_runs/<map>_<time>/
├── <map>_<time>.inputs   ← action sequence (TMInterface format)
├── config.bak.py
└── q_values.joblib
```

---

## 20. TensorBoard Metrics

**Per-race:** race time, zone reached, avg Q-value, action gap, instrumentation timings.

**Aggregated (every 5 min):** loss, gradient norms, learning rate, epsilon, gamma, memory size, learner time budget breakdown, transitions per second, layer L2 norms, IQN quantile spread.

---

## 21. Utilities & Helper Modules

| Module | Key exports |
|---|---|
| `utilities.py` | `soft_copy_param`, `custom_weight_decay`, schedule functions, `save_checkpoint` |
| `contact_materials.py` | Maps TMInterface material IDs → 4 physics behavior categories |
| `reward_shaping.py` | `speedslide_quality_tarmac()` — Numba-JIT speedslide quality (Tomashu's formula) |
| `geometry.py` | Vector math, coordinate transforms |
| `analysis_metrics.py` | Matplotlib debug plots: distribution curves, tau curves, loss distribution |
| `run_to_video.py` | `write_actions_in_tmi_format()` — export `.inputs` for TMInterface replay |

---

## 22. File-by-File Reference

| File | Role |
|---|---|
| `scripts/train.py` | Entry point — spawns collectors, runs learner |
| `config_files/config.py` | Master config (all hyperparameters) |
| `config_files/config_copy.py` | Live config (hot-reloaded during training) |
| `config_files/inputs_list.py` | 12 discrete actions |
| `trackmania_rl/agents/iqn.py` | `IQN_Network`, `Trainer`, `Inferer`, `iqn_loss` |
| `trackmania_rl/multiprocess/learner_process.py` | GPU training loop |
| `trackmania_rl/multiprocess/collector_process.py` | Game-playing rollout workers |
| `trackmania_rl/tmi_interaction/game_instance_manager.py` | `rollout()` — main episode loop |
| `trackmania_rl/tmi_interaction/tminterface2.py` | Low-level socket protocol |
| `trackmania_rl/tmi_interaction/Python_Link.as` | AngelScript plugin injected into TM |
| `trackmania_rl/buffer_management.py` | Rollout → Experience objects |
| `trackmania_rl/buffer_utilities.py` | Buffer collation, mini-race logic, `CustomPrioritizedSampler` |
| `trackmania_rl/map_loader.py` | VCP loading, centerline preprocessing, checkpoint sync |
| `maps/*.npy` | Pre-computed VCP centerline arrays |
| `save/` | Model weights, optimizer state, stats |
| `tensorboard/` | TensorBoard event files |
