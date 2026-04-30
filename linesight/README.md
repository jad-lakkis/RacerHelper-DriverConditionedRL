# Linesight — Complete Technical Documentation

> A deep reinforcement learning agent that learns to drive Trackmania at superhuman speeds.  
> Algorithm: **IQN** (Implicit Quantile Network) · Architecture: Dueling + DDQN · Framework: PyTorch

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Workflow Map](#3-system-workflow-map)
   - [3a. ASCII Workflow Summary](#3a-ascii-workflow-summary)
   - [3b. High-Level Component Architecture](#3b-high-level-component-architecture)
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
23. [Human-Likeness Penalties](#23-human-likeness-penalties)
24. [Braking Aggression Conditioning](#24-braking-aggression-conditioning)

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
│       ├── gbx_to_times_list.py
│       ├── gbx_to_vcp.py
│       ├── tmi1.4.3/
│       ├── tmi2/
│       └── video_stuff/
│
├── config_files/
│   ├── config.py                        # Master config (user-editable)
│   ├── config_copy.py                   # Live config (auto-copied, hot-reloaded)
│   ├── inputs_list.py                   # Action definitions (12 actions)
│   ├── state_normalization.py           # Float input mean/std for normalization
│   └── user_config.py                   # Machine-specific paths & settings
│
├── trackmania_rl/
│   ├── agents/
│   │   └── iqn.py                       # IQN_Network, Trainer, Inferer classes
│   │
│   ├── multiprocess/
│   │   ├── learner_process.py           # GPU training loop + TensorBoard
│   │   ├── collector_process.py         # Game-playing worker processes
│   │   └── debug_utils.py              # Debug helpers
│   │
│   ├── tmi_interaction/
│   │   ├── game_instance_manager.py     # TMInterface bridge — the rollout() function
│   │   ├── tminterface2.py              # Low-level TMInterface socket protocol
│   │   └── Python_Link.as               # AngelScript plugin (injected into the game)
│   │
│   ├── experience_replay/
│   │   └── experience_replay_interface.py  # Experience dataclass
│   │
│   ├── buffer_management.py             # Converts rollouts to Experience objects
│   ├── buffer_utilities.py              # Buffer construction, collate fn, mini-race logic
│   ├── reward_shaping.py                # Speedslide quality reward helper
│   ├── map_loader.py                    # VCP loading, GBX parsing, CP sync
│   ├── map_reference_times.py           # Known author/gold times per map
│   ├── contact_materials.py             # Wheel contact material classifications
│   ├── geometry.py                      # Math helpers
│   ├── analysis_metrics.py              # Plot generation (distribution, tau, etc.)
│   ├── utilities.py                     # NN utils, schedules, checkpointing
│   ├── run_to_video.py                  # Write actions to TMInterface format
│   └── __init__.py
│
├── maps/
│   ├── ESL-Hockolicious_0.5m_cl2.npy   # VCP centerline for Hockolicious map
│   └── map5_0.5m_cl.npy                # VCP centerline for map5
│
├── save/                                # Auto-created; stores weights + stats
├── tensorboard/                         # TensorBoard event files
├── pyproject.toml
├── pixi.toml
├── requirements_pip.txt
└── requirements_conda.txt
```

---

## 3. System Workflow Map

The diagram below shows every file in the repository as a node, color-coded by layer, with labeled arrows for every import, data transfer, or control dependency.

## 3a. ASCII Workflow Summary

For quick reference, here is the simplified linear data flow:

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
│  multiprocess/collector_process.py                                   │
│                                                                      │
│   config_files/config_copy.py ──► reads epsilon, map cycle, etc.    │
│   config_files/inputs_list.py ──► 12 discrete actions               │
│   trackmania_rl/map_loader.py ──► loads .npy VCP centerline         │
│                                                                      │
│   tmi_interaction/game_instance_manager.py                           │
│   │  rollout() ──────────────────────────────────────────────────   │
│   │  │                                                          │   │
│   │  ├─ tmi_interaction/tminterface2.py                         │   │
│   │  │   (socket protocol: reads SC_RUN_STEP_SYNC,             │   │
│   │  │    SC_REQUESTED_FRAME_SYNC, SC_CHECKPOINT_COUNT_CHANGED) │   │
│   │  │                                                          │   │
│   │  ├─ Grabs grayscale frame (160×120 uint8)                   │   │
│   │  ├─ Reads sim state (position, velocity, orientation, etc.) │   │
│   │  ├─ Constructs float feature vector (see §7)                │   │
│   │  ├─ Calls inferer.get_exploration_action()                  │   │
│   │  │   └─ agents/iqn.py  Inferer.infer_network()             │   │
│   │  │       └─ IQN_Network.forward()                           │   │
│   │  ├─ Sends action to game via tminterface2.py                │   │
│   │  └─ Returns rollout_results dict + end_race_stats           │   │
│   │                                                          │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                      │
│   rollout_queue.put((rollout_results, end_race_stats, ...))          │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ (multiprocessing.Queue)
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LEARNER PROCESS (main process)                                      │
│  multiprocess/learner_process.py                                     │
│                                                                      │
│   1. rollout_queues[i].get()  ← receives rollout from collector      │
│                                                                      │
│   2. buffer_management.py                                            │
│      fill_buffer_from_rollout_with_n_steps_rule()                    │
│      │  reward_shaping.py ── speedslide_quality_tarmac()             │
│      │  experience_replay_interface.py ── Experience dataclass       │
│      └─ buffer.extend(experiences)                                   │
│                                                                      │
│   3. buffer_utilities.py                                             │
│      buffer_collate_function()  [called inside buffer.sample()]      │
│      │  Mini-race time horizon sampling                              │
│      │  Reward shaping potential correction                          │
│      └─ Returns GPU tensors ready for training                       │
│                                                                      │
│   4. agents/iqn.py   Trainer.train_on_batch()                        │
│      │  IQN_Network.forward()  ← online_network (state)             │
│      │  IQN_Network.forward()  ← target_network (next_state)        │
│      │  iqn_loss()  ── pinball / quantile Huber loss                │
│      │  scaler.scale(loss).backward()                                │
│      │  clip_grad_norm / clip_grad_value                             │
│      │  RAdam optimizer step                                         │
│      └─ custom_weight_decay                                          │
│                                                                      │
│   5. Every N batches: copy online_network → uncompiled_shared_network│
│      (collectors read from shared_network to stay up-to-date)        │
│                                                                      │
│   6. Every M memories: soft_copy_param(target_network, online, τ)   │
│                                                                      │
│   7. Every 5 min: write TensorBoard scalars, save checkpoint         │
│      utilities.py ── save_checkpoint(), save_run()                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3b. High-Level Component Architecture

A compact, top-down view of every major component and what travels between them:

```
┌───────────────────────────────────┐
│         scripts/train.py          │
│  copies config → config_copy.py   │
│  creates shared memory + queues   │
└────────────────┬──────────────────┘
                 │
      ┌──────────┴──────────┐
      │  spawn via          │  run in main
      │  mp.Process (×N)    │  process
      ▼                     ▼
┌──────────────────┐  ┌─────────────────┐
│ collector_       │  │ learner_        │
│ process.py       │  │ process.py      │
│                  │  │                 │
│ one per game     │  │ GPU training    │
│ instance         │  │ loop            │
└────────┬─────────┘  └────────┬────────┘
         │                     │
  instantiates           instantiates
         │                     │
         ▼                     ▼
┌──────────────────┐  ┌─────────────────┐
│   Inferer        │  │  Trainer        │
│  agents/iqn.py   │  │  agents/iqn.py  │
│                  │  │                 │
│ get_exploration_ │  │ train_on_batch()│
│ action() →       │  │ IQN pinball     │
│ ε-greedy /       │  │ loss + RAdam    │
│ Boltzmann /      │  │ optimizer step  │
│ greedy argmax    │  └────────┬────────┘
└────────┬─────────┘           │
         │                     │
  action selection             │
  (called inside rollout)      │
         │                     │
         ▼                     │
┌──────────────────────┐       │
│ GameInstanceManager  │       │
│                      │       │
│ rollout() — main     │       │
│ per-episode loop;    │       │
│ grabs 160×120 frame, │       │
│ builds float vec     │       │
│ (188,), tracks VCPs  │       │
└──────────┬───────────┘       │
           │                   │
    SC_RUN_STEP_SYNC /         │
    SC_REQUESTED_FRAME_SYNC    │
    (local socket, 50 ms tick) │
           │                   │
           ▼                   │
┌──────────────────────┐       │
│ TMInterface          │       │
│ tminterface2.py      │       │
│                      │       │
│ low-level socket     │       │
│ read / write /       │       │
│ respond protocol     │       │
└──────────┬───────────┘       │
           │                   │
    exposes sim state          │
    via AngelScript bridge     │
           │                   │
           ▼                   │
┌──────────────────────┐       │
│ Python_Link.as       │       │
│ (AngelScript plugin) │       │
│                      │       │
│ injected into TM at  │       │
│ startup; streams     │       │
│ position / velocity  │       │
│ / orientation        │       │
└──────────┬───────────┘       │
           │                   │
    runs inside                │
    game process               │
           │                   │
           ▼                   │
┌──────────────────────┐       │
│ Trackmania Forever   │       │
│ (game simulation)    │       │
└──────────┬───────────┘       │
           │                   │
           │  rollout_results  │
           │  via mp.Queue:    │
           │  • frames (uint8) │
           │  • actions        │
           │  • float vecs     │
           │  • Q-values       │
           │  • zone indices   │
           └──────────────────►│
                               ▼
                    ┌──────────────────────┐
                    │   buffer_management  │
                    │                      │
                    │ fill_buffer_from_    │
                    │ rollout_with_n_steps │
                    │ • compute per-step   │
                    │   rewards (time pen  │
                    │   + VCP progress)    │
                    │ • build n-step (n=3) │
                    │   discounted returns │
                    │ • create Experience  │
                    │   dataclass objects  │
                    └──────────┬───────────┘
                               │
                      buffer.extend(experiences)
                               │
                               ▼
                    ┌──────────────────────┐
                    │    replay buffer     │
                    │  (CustomPrioritized  │
                    │    Sampler)          │
                    │                      │
                    │ buffer_utilities.py  │
                    │ • mini-race horizon  │
                    │   sampling (7 s)     │
                    │ • overwrite          │
                    │   state_float[:,0]=t │
                    │ • potential-based    │
                    │   reward shaping     │
                    └──────────┬───────────┘
                               │
                   buffer.sample() → collate_fn
                   → GPU tensors (float16 img,
                     float32 floats, actions,
                     rewards, gammas, terminals)
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Trainer.train_on_    │
                    │ batch()              │
                    │                      │
                    │ • online_net forward │
                    │   (state, τ₃)        │
                    │ • target_net forward │
                    │   (next_state, τ₂)   │
                    │ • iqn_loss() pinball │
                    │   Huber across τ²×τ³ │
                    │ • self-loss norm     │
                    │ • scaler.backward()  │
                    │ • clip_grad_norm(30) │
                    │ • RAdam step         │
                    └──────────┬───────────┘
                               │
                  every 10 batches, copy weights
                  to shared memory (under lock)
                               │
                               ▼
                    ┌──────────────────────┐
                    │    shared_network    │
                    │                      │
                    │  learner writes ──►  │
                    │                      │
                    │  collectors pull ◄── │
                    │  (every 20 actions,  │
                    │   under lock)        │
                    └──────────────────────┘
```

---

## 4. Multiprocessing Architecture

The system uses Python `torch.multiprocessing` with the following shared objects:

| Object | Type | Purpose |
|---|---|---|
| `uncompiled_shared_network` | `IQN_Network` in shared memory | Bridge to pass updated weights from learner → collectors |
| `shared_network_lock` | `Lock` | Prevents race condition when reading/writing network weights |
| `game_spawning_lock` | `Lock` | Serializes game launch so two processes don't open games simultaneously |
| `shared_steps` | `mp.Value(c_int64)` | Atomic counter for total frames played, used to compute epsilon schedules |
| `rollout_queues` | `[mp.Queue]` | One queue per collector; delivers completed rollout data to the learner |

### Process count
`config.gpu_collectors_count` (default: `2`) determines how many parallel game instances run. The learner runs in the main process to avoid creating an extra CUDA context.

### Network weight sharing
1. Learner trains `online_network` (compiled/JIT).
2. Every `send_shared_network_every_n_batches` (default: 10) batches, learner copies `uncompiled_online_network.state_dict()` → `uncompiled_shared_network` (under lock).
3. Each collector calls `update_network()` every `update_inference_network_every_n_actions` (default: 20) actions, pulling from `uncompiled_shared_network` (under lock) into its local `uncompiled_inference_network`, then into the compiled `inference_network`.

---

## 5. Configuration System

**Files:** `config_files/config.py`, `config_files/config_copy.py`

Two config files coexist:

- `config.py` — the user-facing master config, tracked in git. Modifying this file during a run has **no effect** on that run.
- `config_copy.py` — a live copy created at startup (`shutil.copyfile`). Both the learner and collectors call `importlib.reload(config_copy)` on every iteration, so **editing `config_copy.py` during training applies changes on the fly** without clearing the replay buffer or restarting.

This design enables:
- Safe git operations and code changes without disrupting a running experiment.
- Real-time tuning of learning rate, exploration, reward shaping weights, etc.

### Key config sections

| Section | Variables |
|---|---|
| Image input | `W_downsized=160`, `H_downsized=120` |
| Timing | `tm_engine_step_per_action=5` (50ms per action), `ms_per_action=50` |
| VCP geometry | `n_zone_centers_in_inputs=40`, `one_every_n_zone_centers_in_inputs=20` |
| Float feature dim | `float_input_dim = 27 + 3×40 + 4×5 + 4×4 + 1 = 188` |
| Network sizes | `float_hidden_dim=256`, `conv_head_output_dim=5632`, `dense_hidden_dimension=1024`, `iqn_embedding_dimension=64` |
| IQN | `iqn_n=8` (training quantiles), `iqn_k=32` (inference quantiles), `iqn_kappa=5e-3` |
| Replay | `number_times_single_memory_is_used_before_discard=32` |
| Optimization | `batch_size=512`, `adam_beta1=0.9`, `adam_beta2=0.999`, `adam_epsilon=1e-4` |
| Mini-race | `temporal_mini_race_duration_ms=7000` → 140 actions at 50ms |
| Cutoffs | `cutoff_rollout_if_race_not_finished_within_duration_ms=300_000` |

---

## 6. Game Interaction Layer

**Files:** `trackmania_rl/tmi_interaction/game_instance_manager.py`, `tminterface2.py`, `Python_Link.as`

### TMInterface (TMI)
Linesight hooks into Trackmania Nations Forever via **TMInterface 2**, a modding framework that exposes the game's simulation state over a local socket. The AngelScript plugin `Python_Link.as` is injected at startup.

### `GameInstanceManager.rollout()`
This is the main per-episode function. It:

1. **Launches / reconnects** to the game if needed.
2. **Loads the map** via `iface.execute_command("map <path>")`.
3. **Rewinds to a saved start state** (`iface.rewind_to_state`) for fast episode restarts without reloading.
4. **Event loop** — reads incoming `MessageType` messages from TMI:
   - `SC_RUN_STEP_SYNC`: fires every `tm_engine_step_per_action × 10 ms = 50ms`. At every action step, the game is paused (`request_speed(0)`), the simulation state is read, a frame is requested, then an action is computed and applied before unpausing.
   - `SC_REQUESTED_FRAME_SYNC`: the grabbed `160×120` BGRA frame is converted to grayscale, passed to the exploration policy, and the chosen action is sent back.
   - `SC_CHECKPOINT_COUNT_CHANGED_SYNC`: detects race finish. The finish is deliberately **prevented** (`cp_times[-1].time = -1`) to allow the episode to continue cleanly and record statistics.
5. **Termination** conditions:
   - Race finished (checkpoint count == target count).
   - `max_overall_duration_ms` elapsed (300 seconds by default).
   - No VCP progress for `max_minirace_duration_ms` (2 seconds).
6. Returns `rollout_results` (frames, actions, floats, q-values, zone indices) and `end_race_stats`.

### Virtual Zone Index Tracking
`update_current_zone_idx()` (Numba-JIT compiled) advances the current VCP index whenever the car is closer to the next VCP than the current one, subject to:
- Distance to virtual checkpoint ≤ `max_allowable_distance_to_virtual_checkpoint`.
- The car has actually passed near the corresponding real checkpoint (anti-cut measure).

---

## 7. State Space — What the Agent Sees

Each decision step provides **two inputs** to the neural network:

### 7.1 Image input — `(1, 120, 160)` uint8

A **grayscale** screenshot of the game at `160×120` resolution, captured via TMInterface's frame buffer. Pixel values are in `[0, 255]`. During training they are normalized to `[-1, 1]` via `(img.float() - 128) / 128`.

The game UI is disabled during rollouts (`toggle_interface(False)`) so the agent only sees the road and environment.

### 7.2 Float feature vector — shape `(188,)` float32

All features are expressed **in the car's local reference frame** (rotated by the car's orientation matrix). The full vector is:

| Index | Feature | Description |
|---|---|---|
| 0 | `temporal_mini_race_current_time_actions` | How far into the 7-second mini-race window we are (steps). Injected at collation time, not during rollout. |
| 1–20 | Previous 5 actions | 4 binary flags per action (accelerate, brake, left, right) × 5 steps |
| 21–36 | Wheel & gear state | per-wheel: sliding flag, ground contact, damper absorb; gearbox state, gear number, RPM, gear counter |
| 37–52 | Contact material types | 4 physics behavior types × 4 wheels (one-hot style) |
| 53–55 | Angular velocity | Car angular velocity in car frame (3D) |
| 56–58 | Linear velocity | Car velocity in car frame (3D); index 56 = lateral, 58 = forward |
| 59–61 | Map up-vector | The world Y axis expressed in car frame |
| 62–181 | VCP coordinates | 40 upcoming virtual checkpoint positions relative to car, sampled every 20th VCP = look-ahead of ~200m |
| 182 | Distance to finish | Meters remaining to the finish line (capped at `margin_to_announce_finish_meters = 700`) |
| 183 | `is_freewheeling` | Boolean: engine disconnected from wheels |

**Normalization:** The float feature extractor subtracts `float_inputs_mean` and divides by `float_inputs_std` (defined in `state_normalization.py`) as the very first operation inside `IQN_Network.forward()`.

---

## 8. Action Space — What the Agent Can Do

**File:** `config_files/inputs_list.py`

12 discrete actions, each defined as a combination of four binary flags: `{accelerate, brake, left, right}`.

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

Actions are applied for exactly one step (50ms) each.

---

## 9. Reward Function

**Files:** `buffer_management.py`, `reward_shaping.py`, `config_files/config.py`

The reward for step `i → i+1` is computed in `fill_buffer_from_rollout_with_n_steps_rule()`:

### Core reward components

```
reward[i] = constant_reward_per_ms × ms_per_action
           + reward_per_m_advanced_along_centerline × (meters[i] - meters[i-1])
```

Default values: `constant_reward_per_ms = -6/5000` (penalty for time), `reward_per_m_advanced_along_centerline = 5/500` (progress reward). These are tuned so that advancing 1 meter along the track exactly compensates for the time cost of traveling that meter at the slowest viable speed.

### Optional engineered rewards (all default to 0)

These are annealed via linear schedules and intended for curriculum purposes:

| Variable | Trigger | Purpose |
|---|---|---|
| `engineered_speedslide_reward` | All 4 wheels grounded | Reward perfect speedslide quality (= 1.0 from Tomashu's formula) |
| `engineered_neoslide_reward` | Lateral speed ≥ 2 m/s | Reward any lateral sliding |
| `engineered_kamikaze_reward` | Action ≤ 2 OR ≤1 wheel grounded | Reward risky maneuvers |
| `engineered_close_to_vcp_reward` | Always | Reward staying close to the next VCP |

### Reward potential shaping
At collation time, the reward is augmented with potential-based shaping (Ng et al., 1999):

```
adjusted_reward = raw_reward + γ × Φ(next_state) − Φ(state)
```

where `Φ(state)` is computed in `get_potential()`:

```python
Φ(state) = shaped_reward_dist_to_cur_vcp × clip(||state[62:65]||, min_dist, max_dist)
           + shaped_reward_point_to_vcp_ahead × (vcp_direction_z - 1)
```

This encourages the car to stay close to the upcoming VCP without changing the optimal policy.

---

## 10. Map & Virtual Checkpoint System

**File:** `trackmania_rl/map_loader.py`, `maps/*.npy`

### Virtual Checkpoints (VCPs)
VCPs are pre-computed 3D points along the track centerline, spaced 0.5m apart and stored as `.npy` files (e.g., `ESL-Hockolicious_0.5m_cl2.npy`). They define the reference line that the agent should follow.

The `.npy` file is generated offline from a ghost replay (`.Replay.gbx`) via `scripts/tools/gbx_to_vcp.py`, which:
1. Parses the ghost's position records with `pygbx`.
2. Interpolates the positions to 0.5m spacing using `make_interp_spline`.
3. Saves the resulting `(N, 3)` array.

### Loading and preprocessing
`load_next_map_zone_centers()` in `map_loader.py`:
1. Loads the `.npy` file.
2. Prepends `n_zone_centers_extrapolate_before_start_of_map=20` artificial zones before the start line.
3. Appends `n_zone_centers_extrapolate_after_end_of_map=1000` artificial zones past the finish.
4. Smooths the trajectory: `zone_centers[5:-5] = 0.5 × (zone_centers[:-10] + zone_centers[10:])`.

### Precalculated geometry
`precalculate_virtual_checkpoints_information()` computes:
- `zone_transitions`: midpoints between consecutive VCPs.
- `distance_between_zone_transitions`: segment lengths.
- `distance_from_start_track_to_prev_zone_transition`: cumulative distance from start.
- `normalized_vector_along_track_axis`: unit vectors along each segment.

These are used every step to compute `meters_advanced_along_centerline` (the progress signal).

### Real checkpoint synchronization
`sync_virtual_and_real_checkpoints()` reads the actual checkpoint block positions from the `.gbx` challenge file, then matches each real checkpoint to its closest VCP. During rollout, `update_current_zone_idx()` requires the car to have been within 12m of the real checkpoint center before it can advance past the corresponding VCPs. This prevents the agent from finding shortcuts that skip checkpoints.

---

## 11. Neural Network Architecture

**File:** `trackmania_rl/agents/iqn.py` — `IQN_Network`

The network has two input heads whose outputs are concatenated, then mixed with a quantile embedding before being fed to dueling output heads.

```
Image (1×120×160, float16)
   │
   ├── Conv2D(1→16, 4×4, stride=2)   LeakyReLU
   ├── Conv2D(16→32, 4×4, stride=2)  LeakyReLU
   ├── Conv2D(32→64, 3×3, stride=2)  LeakyReLU
   ├── Conv2D(64→32, 3×3, stride=1)  LeakyReLU
   └── Flatten  →  conv_output (5632,)

Float features (188, float32)
   │  [normalized: (x - mean) / std]
   ├── Linear(188→256)  LeakyReLU
   └── Linear(256→256)  LeakyReLU  →  float_output (256,)

concat = cat(conv_output, float_output)  →  (5888,)

Quantile τ ~ U[0,1]  (batch × iqn_n)
   ├── cos(π × i × τ) for i=1..64   →  (batch×n_quantiles, 64)
   ├── Linear(64→5888)
   └── LeakyReLU  →  quantile_embedding (5888,)

combined = concat × quantile_embedding  [Hadamard product]  →  (5888,)

Advantage head A:
   ├── Linear(5888→512)  LeakyReLU
   └── Linear(512→12)    →  A(s, a, τ)

Value head V:
   ├── Linear(5888→512)  LeakyReLU
   └── Linear(512→1)     →  V(s, τ)

Q(s, a, τ) = V(s, τ) + A(s, a, τ) − mean_a[A(s, a, τ)]
```

### Weight initialization
All convolutional and linear layers use **orthogonal initialization** with gain calculated for LeakyReLU (slope 0.01). The IQN embedding layer uses `√2 × activation_gain` to compensate for the reduced variance of cosine inputs.

### Compilation
- On **Linux**: `torch.compile()` with `"max-autotune"` mode for inference and `"max-autotune-no-cudagraphs"` for training.
- On **Windows**: `torch.jit.script()`.
- Compilation is controlled by `config.use_jit = True`.

Two copies of the network are maintained: a JIT-compiled version used for all computation, and an **uncompiled version** (`uncompiled_online_network`) used only for weight sharing across processes (compiled networks cannot be serialized to shared memory).

---

## 12. IQN Algorithm — Theory & Implementation

**File:** `trackmania_rl/agents/iqn.py`

### Background
IQN (Implicit Quantile Network, Dabney et al. 2018) learns the **full return distribution** Z(s,a) rather than just its expectation E[Z] = Q(s,a). For any quantile τ ∈ (0,1), the network outputs the τ-th quantile of the return distribution. At decision time, the agent averages across many quantile samples to get a robust estimate of Q.

### Quantile sampling (forward pass)

```python
# Symmetric sampling around 0.5 for variance reduction:
tau_half = (arange(iqn_n//2) + rand(batch × iqn_n//2)) / iqn_n
tau = cat(tau_half, 1 - tau_half)  # shape: (batch × iqn_n, 1)

# Cosine embedding of quantile:
quantile_embedding = cos(π × [1..64] × tau)       # (batch×n, 64)
quantile_embedding = Linear(64→5888)(quantile_embedding)  # (batch×n, 5888)

# Hadamard product with state embedding:
combined = state_embedding.repeat(n_quantiles, 1) × quantile_embedding
```

### IQN Loss (Quantile Huber / Pinball loss)

For a batch of transitions, we sample `iqn_n` quantiles for both the target and the prediction:

```
TD_error[i,j] = target_τ2[i] − output_τ3[j]

loss[i,j] = Huber_κ(TD_error[i,j]) × |τ3[j] − 𝟙[TD_error<0]|
```

where `Huber_κ(x) = x²/(2κ) if |x|<κ else |x| - κ/2` with `κ = iqn_kappa = 5e-3`.

This is the **pinball loss** that pushes each quantile prediction to be the correct quantile of the return distribution.

**Implementation:**
```python
def iqn_loss(targets, outputs, tau_outputs, num_quantiles, batch_size):
    TD_error = targets[:, :, None, :] - outputs[:, None, :, :]
    # (batch, iqn_n_target, iqn_n_output, 1)
    loss = Huber_kappa(TD_error)
    tau = tau_outputs.reshape([num_quantiles, batch_size, 1]).transpose(0,1)
    tau = tau[:, None, :, :].expand([-1, num_quantiles, -1, -1])
    loss = (where(TD_error < 0, 1-tau, tau) * loss).sum(dim=2).mean(dim=1)[:, 0]
    return loss  # (batch_size,)
```

### DDQN variant
When `use_ddqn = True` (default: `False`): the **online network** selects the best action for the next state (averaged across quantiles), and the **target network** evaluates that action. This decouples action selection from evaluation, reducing overestimation bias.

### Self-loss normalization
A novel training-stability trick: the loss is normalized by the "self-loss" — the IQN loss when the target equals the output itself (measuring how spread the distribution is):

```python
target_self_loss = sqrt(iqn_loss(targets, targets, tau2, ...))
typical_self_loss = 0.99 × typical_self_loss + 0.01 × target_self_loss.mean()
correction = target_self_loss.clamp(min=typical_self_loss / clamp_ratio)
loss = loss × (typical_clamped_self_loss / correction)
```

This prevents transitions with extremely broad return distributions from dominating the gradient.

---

## 13. Replay Buffer & Mini-Race Logic

**Files:** `buffer_utilities.py`, `buffer_management.py`, `experience_replay/experience_replay_interface.py`

### Experience dataclass
Each stored transition (`Experience`) contains:
- `state_img` — `(1, H, W)` uint8
- `state_float` — `(188,)` float32
- `state_potential` — float (for reward shaping)
- `action` — int
- `n_steps` — how many steps between state and next_state (up to `n_steps_max=3`)
- `rewards` — `(n_steps_max,)` float32: cumulative discounted reward at each possible step count
- `next_state_img`, `next_state_float`, `next_state_potential`
- `gammas` — `(n_steps_max,)` float32: `γ^1, γ^2, ..., γ^n`
- `terminal_actions` — distance to race finish in steps (or `inf` if race didn't finish)

### N-step returns
`fill_buffer_from_rollout_with_n_steps_rule()` builds transitions with multi-step returns (default `n_steps=3`):

```
reward[i] = Σ_{j=0}^{n-1} γ^j × r[i+j+1]
```

When `discard_non_greedy_actions_in_nsteps = True`, n is capped at the first non-greedy action in the lookahead. This prevents the n-step estimate from including rewards from exploratory actions.

### Mini-race / clipped horizon logic
This is the most architecturally novel component. Rather than optimizing the full episode return (which can be thousands of steps), Linesight defines Q-values over a **7-second window** only.

The key insight: if we define Q(s,a) = "total reward over the next 7 seconds starting from state s taking action a", then we can optimize with `γ = 1` (no discounting within the 7-second window), which simplifies learning.

**At collation time** (`buffer_collate_function()`):
1. For each sampled transition, a random "mini-race start time" `t` is drawn in `[0, 140]` steps (with oversampling of short horizons).
2. `state_float[:, 0]` is **overwritten** with `t` (the time within the mini-race).
3. `next_state_float[:, 0]` is overwritten with `t + n_steps`.
4. If `t + n_steps ≥ 140` (mini-race would end), the transition is treated as **terminal** (gamma set to 0).
5. This allows the network to condition its Q-values on time remaining, learning a different value function at each time step within the window.

This trick lets the agent use `γ = 1` while avoiding the instability of infinite-horizon discounted returns.

### Buffer sizing
Buffer size grows via a staircase schedule:

| Frames | Buffer size | Start learning at |
|---|---|---|
| 0 | 50,000 | 20,000 |
| 5M | 100,000 | 75,000 |
| 7M | 200,000 | 150,000 |

### Prioritized Experience Replay
`CustomPrioritizedSampler` extends torchrl's `PrioritizedSampler`:
- Default priority for new memories = `default_priority_ratio × average_priority` (instead of max priority), giving new memories moderate sampling weight.
- Priority update uses TD error: `|mean_τ[Q(s,a,τ)] - target_mean|`.
- Only transitions with `state_float[0] >= temporal_mini_race_duration_actions - 40` update priorities (transitions near mini-race end are skipped, as their targets are less meaningful).
- `prio_alpha=0` by default (uniform sampling).

---

## 14. Training Loop — Step by Step

**File:** `multiprocess/learner_process.py`

```
while True:
    1. Wait for rollout from any collector queue
    
    2. importlib.reload(config_copy)   ← hot-reload config changes
    
    3. Compute current schedule values:
       - learning_rate   (exponential schedule)
       - gamma           (linear schedule)
       - engineered_*_reward  (linear schedules)
       - epsilon, epsilon_boltzmann  (exponential schedules)
       - weight_decay = weight_decay_lr_ratio × lr
    
    4. Update optimizer hyperparameters in-place
    
    5. (Optional) Plot analysis curves every ~85 eval runs
    
    6. Accumulate frame count
    
    7. Write per-race TensorBoard scalars
    
    8. Save best run if this race set a new record
    
    9. if fill_buffer:
       a. Convert rollout → Experience objects (buffer_management.py)
       b. Add to replay buffer
       c. Check if periodic network reset is due
       
       d. TRAINING LOOP:
          while buffer has enough memories AND we haven't trained enough:
             - Sample batch (triggers buffer_collate_function)
             - Trainer.train_on_batch(buffer, do_learn=True)
               * Forward pass: online_network(state)    → Q(s,a,τ3)
               * Forward pass: target_network(next)     → Q(s',a',τ2)
               * (if DDQN) online_network(next)         → best action
               * Compute IQN loss
               * Compute self-loss normalization
               * Weighted sum with IS weights (if PER)
               * Backward + clip gradients
               * Optimizer step
             - custom_weight_decay(online_network, 1 - weight_decay)
             - Every 10 batches: push weights to shared_network
             - Every 2048 memories: soft_copy_param(target ← online, τ=0.02)
    
    10. Every 5 minutes:
        - Write aggregated stats to TensorBoard
        - Compute IQN spread across quantiles
        - Print buffer mean/std
        - (If PER) print priority distribution
        - save_checkpoint(weights1.torch, weights2.torch, optimizer1.torch, scaler.torch)
        - joblib.dump(accumulated_stats)
```

### Memory usage accounting
The learner tracks `cumul_number_single_memories_used` and `cumul_number_single_memories_should_have_been_used`. The difference controls how many training batches to run:

```
should_have_been_used += number_times_single_memory_is_used_before_discard × new_memories
```

With `number_times_single_memory_is_used_before_discard = 32` and `batch_size = 512`, each new memory will be trained on roughly 32 times before being phased out.

---

## 15. Exploration Strategy

**File:** `trackmania_rl/agents/iqn.py` — `Inferer.get_exploration_action()`

Three layered exploration mechanisms:

### 1. Epsilon-greedy
With probability `ε`, select a uniformly random action. `ε` decays exponentially:

```
(0 frames: 1.0) → (50k: 1.0) → (300k: 0.1) → (3M: 0.03)
```

### 2. Boltzmann / softmax exploration
With probability `ε_boltzmann`, add Gaussian noise to Q-values before argmax:

```python
action = argmax(Q(s) + τ_boltzmann × randn(12))
```

`ε_boltzmann` decays from 0.15 → 0.03. `τ_boltzmann = 0.01` (controls noise scale).

### 3. Greedy
Otherwise: `action = argmax(mean_τ[Q(s,a,τ)])`.

### Exploration vs. evaluation
Each map in the cycle is marked `is_explo` (True/False):
- Exploration runs use epsilon-greedy + Boltzmann, and the network runs in `.train()` mode (batch norm tracks stats if present).
- Evaluation runs are purely greedy (epsilon=0, Boltzmann=0), network in `.eval()` mode.
- By default: 4 exploration runs followed by 1 evaluation run per map in the cycle.

---

## 16. Target Network & Soft Updates

**File:** `multiprocess/learner_process.py`, `utilities.py`

### Soft update
Every `number_memories_trained_on_between_target_network_updates = 2048` memories trained on:

```python
target_network ← (1 - τ) × target_network + τ × online_network
```

with `soft_update_tau = 0.02`. This is implemented via `soft_copy_param()` which does an in-place linear combination of all parameter tensors.

### Periodic network reset
Controlled by `reset_every_n_frames_generated` (default effectively disabled at `400_000_00000000`). When triggered:
1. A fresh untrained network is created.
2. `online_network ← (1 - overall_reset_mul_factor) × online_network + overall_reset_mul_factor × untrained` (very gentle perturbation, `mul_factor=0.01`).
3. The last layer (A and V heads' final linear) is reset more aggressively: `last_layer_reset_factor=0.8` blends toward untrained weights.
4. `additional_transition_after_reset = 1_600_000` extra training steps are added to the budget.

This implements a form of "primacy bias" mitigation (Nikishin et al., 2022).

---

## 17. Hyperparameter Reference

| Hyperparameter | Value | Description |
|---|---|---|
| `iqn_n` | 8 | Number of quantiles sampled during training (N in IQN paper) |
| `iqn_k` | 32 | Number of quantiles sampled during inference (K in IQN paper) |
| `iqn_kappa` | 5e-3 | Huber loss threshold in quantile loss |
| `iqn_embedding_dimension` | 64 | Dimension of cosine quantile embedding |
| `batch_size` | 512 | Transitions per training batch |
| `n_steps` | 3 | Maximum n-step return lookahead |
| `gamma_schedule` | 0.999→1.0 | Discount factor (annealed from 0.999 to 1.0) |
| `soft_update_tau` | 0.02 | Polyak averaging coefficient |
| `clip_grad_norm` | 30 | Maximum gradient L2 norm |
| `clip_grad_value` | 1000 | Maximum per-parameter gradient value |
| `adam_beta1` | 0.9 | RAdam β₁ |
| `adam_beta2` | 0.999 | RAdam β₂ |
| `adam_epsilon` | 1e-4 | RAdam ε (numerical stability) |
| `lr_schedule` | 1e-3 → 5e-5 → 1e-5 | Learning rate schedule |
| `weight_decay_lr_ratio` | 1/50 | Weight decay = lr / 50 |
| `temporal_mini_race_duration_ms` | 7000 | Mini-race horizon (7 seconds) |
| `running_speed` | 80× | Game simulation speed multiplier |
| `update_inference_network_every_n_actions` | 20 | Collectors refresh weights every 20 actions |
| `send_shared_network_every_n_batches` | 10 | Learner pushes weights every 10 batches |
| `target_self_loss_clamp_ratio` | 4 | Clamp ratio for self-loss normalization |

---

## 18. Scheduling System

**File:** `trackmania_rl/utilities.py`

Three schedule types, all keyed on total frames played (`cumul_number_frames_played`):

### Exponential schedule (`from_exponential_schedule`)
Interpolates exponentially between setpoints. Used for: `lr_schedule`, `epsilon_schedule`, `epsilon_boltzmann_schedule`.

```python
value(t) = begin_value × exp(-log(begin/end) × (t - t_begin) / period)
```

### Linear schedule (`from_linear_schedule`)
`numpy.interp` between setpoints. Used for: `gamma_schedule`, all engineered reward schedules.

### Staircase schedule (`from_staircase_schedule`)
Jumps at setpoints, no interpolation. Used for: `memory_size_schedule`, `tensorboard_suffix_schedule`.

---

## 19. Checkpointing & Saving

**Files:** `utilities.py` — `save_checkpoint()`, `save_run()`

### Periodic checkpoint (every 5 minutes)
```
save/
└── <run_name>/
    ├── weights1.torch         ← online_network.state_dict()
    ├── weights2.torch         ← target_network.state_dict()
    ├── optimizer1.torch       ← RAdam state
    ├── scaler.torch           ← AMP GradScaler state
    └── accumulated_stats.joblib  ← all counters, alltime_min_ms, rolling_mean_ms
```

### Best run checkpoint
When a new all-time best race time is set (after `frames_before_save_best_runs = 1.5M` frames):
```
save/<run_name>/best_runs/<map>_<time>/
    ├── <map>_<time>.inputs   ← action sequence in TMInterface format
    ├── config.bak.py         ← config snapshot
    └── q_values.joblib       ← per-step Q-values
    weights1.torch, weights2.torch, optimizer1.torch, scaler.torch
```

### Good runs
All runs below `threshold_to_save_all_runs_ms` (default: disabled at -1) are saved with just the `.inputs` file.

### Resume behavior
On startup, both `learner_process.py` and `collector_process.py` attempt to load weights from `save/<run_name>/weights1.torch`. Training resumes seamlessly.

---

## 20. TensorBoard Metrics

**File:** `multiprocess/learner_process.py`

### Per-race metrics (logged after every rollout)
- `race_time_ratio_<map>` — actual race time / wall-clock rollout time
- `explo_race_time_<status>_<map>` / `eval_race_time_<status>_<map>` — race time in seconds
- `eval_race_time_robust_<status>_<map>` — race time, only logged if within 2% of rolling mean (filters outliers)
- `eval_ratio_<status>_<reference>_<map>` — ratio vs. author/gold time × 100
- `single_zone_reached_<status>_<map>` — furthest VCP reached
- `avg_Q_<status>_<map>` — mean Q-value across the episode
- `mean_action_gap` — mean advantage gap between best and chosen action
- Instrumentation timings: `grab_frame`, `grab_floats`, `exploration_policy`, etc.
- `q_value_0_starting_frame` — Q-value at the very start of the race

### Aggregated metrics (logged every 5 minutes)
- `loss`, `loss_test` — mean IQN loss on train/test buffer
- `grad_norm_history_d9`, `grad_norm_history_d98` — gradient norm percentiles
- `gamma`, `epsilon`, `learning_rate`, `weight_decay`, `memory_size`, etc.
- `learner_percentage_waiting` / `_training` / `_testing` — time budget breakdown
- `transitions_learned_per_second` — throughput
- Layer L2 norms and optimizer state (exp_avg, exp_avg_sq) for each named parameter
- `std_within_iqn_quantiles_for_action{i}` — spread of quantile predictions
- Priority distribution stats (min, q1, mean, median, q3, d9, d98, max) if PER enabled

---

## 21. Utilities & Helper Modules

### `trackmania_rl/utilities.py`
- `init_orthogonal` / `init_kaiming` / `init_xavier` — weight initialization helpers.
- `soft_copy_param` — Polyak averaging between networks.
- `custom_weight_decay` — multiplies all parameters by `(1 - wd)` directly (no L2 in optimizer).
- Schedule functions: `from_exponential_schedule`, `from_linear_schedule`, `from_staircase_schedule`.
- `count_parameters` — pretty-prints parameter table.
- `save_checkpoint`, `save_run` — persistence.
- `set_random_seed` — seeds Python, NumPy, and PyTorch globally.

### `trackmania_rl/contact_materials.py`
Maps TMInterface's integer contact material IDs to one of 4 physics behavior categories (road, dirt, grass, other). Used in the float feature vector (wheel contact material one-hot).

### `trackmania_rl/reward_shaping.py`
`speedslide_quality_tarmac(speed_x, speed_z)` — Numba-JIT function computing the quality of a speedslide maneuver. Quality = 1.0 is perfect; < 1 means understeer; > 1 means oversteer with speed loss. Based on Tomashu's documented friction model.

### `trackmania_rl/geometry.py`
General geometric utility functions (vector math, coordinate transforms).

### `trackmania_rl/analysis_metrics.py`
Generates matplotlib plots for debugging: `distribution_curves`, `race_time_left_curves`, `tau_curves`, `loss_distribution`, `highest_prio_transitions`. These are written to `save/<run_name>/` on demand.

### `trackmania_rl/run_to_video.py`
`write_actions_in_tmi_format()` — converts the `rollout_results["actions"]` list to a `.inputs` file readable by TMInterface, enabling replay of the best runs in-game.

### `trackmania_rl/map_reference_times.py`
Dictionary of known author and gold medal times per map name, used to compute `eval_ratio` metrics in TensorBoard.

### `scripts/tools/`
- `gbx_to_vcp.py` — offline tool to generate a `.npy` centerline from a ghost replay.
- `gbx_to_times_list.py` — extracts checkpoint split times from a replay.
- `tmi2/add_cp_as_triggers.py`, `add_vcp_as_triggers.py` — debug tools to visualize checkpoints/VCPs in-game.
- `video_stuff/animate_race_time.py`, `a_v_widget_folder.py` — video overlay generation.
- `video_stuff/inputs_to_gbx.py` — converts `.inputs` file back to a playable ghost.

---

## 22. File-by-File Reference

| File | Role | Key exports |
|---|---|---|
| `scripts/train.py` | **Entry point** | Spawns collectors, runs learner |
| `config_files/config.py` | **Master config** | All hyperparameters |
| `config_files/config_copy.py` | **Live config** | Hot-reloaded during training |
| `config_files/inputs_list.py` | **Action space** | `inputs` list of 12 actions |
| `config_files/state_normalization.py` | **Normalization** | `float_inputs_mean`, `float_inputs_std` |
| `config_files/user_config.py` | **Machine config** | Paths to game, TMI, OS flag |
| `trackmania_rl/agents/iqn.py` | **RL algorithm** | `IQN_Network`, `Trainer`, `Inferer`, `iqn_loss` |
| `trackmania_rl/multiprocess/learner_process.py` | **Training loop** | `learner_process_fn` |
| `trackmania_rl/multiprocess/collector_process.py` | **Rollout worker** | `collector_process_fn` |
| `trackmania_rl/tmi_interaction/game_instance_manager.py` | **Game bridge** | `GameInstanceManager`, `rollout()` |
| `trackmania_rl/tmi_interaction/tminterface2.py` | **Socket protocol** | `TMInterface`, `MessageType` |
| `trackmania_rl/tmi_interaction/Python_Link.as` | **Game plugin** | AngelScript injected into TM |
| `trackmania_rl/experience_replay/experience_replay_interface.py` | **Transition type** | `Experience` dataclass |
| `trackmania_rl/buffer_management.py` | **Rollout → buffer** | `fill_buffer_from_rollout_with_n_steps_rule` |
| `trackmania_rl/buffer_utilities.py` | **Buffer ops** | `buffer_collate_function`, `make_buffers`, `CustomPrioritizedSampler` |
| `trackmania_rl/reward_shaping.py` | **Reward shaping** | `speedslide_quality_tarmac` |
| `trackmania_rl/map_loader.py` | **Map processing** | `load_next_map_zone_centers`, `precalculate_virtual_checkpoints_information`, `sync_virtual_and_real_checkpoints` |
| `trackmania_rl/map_reference_times.py` | **Reference times** | `reference_times` dict |
| `trackmania_rl/contact_materials.py` | **Material IDs** | `physics_behavior_fromint` |
| `trackmania_rl/utilities.py` | **Utilities** | Schedule functions, save/load, weight init |
| `trackmania_rl/analysis_metrics.py` | **Debug plots** | `distribution_curves`, `tau_curves`, etc. |
| `trackmania_rl/run_to_video.py` | **Export** | `write_actions_in_tmi_format` |
| `trackmania_rl/geometry.py` | **Math** | Geometric helpers |
| `maps/*.npy` | **Track data** | Pre-computed VCP centerline arrays |
| `save/` | **Persistence** | Weights, optimizer state, stats |
| `tensorboard/` | **Monitoring** | TensorBoard event files |

---

---

## 23. Human-Likeness Penalties

**Files:** `trackmania_rl/buffer_management.py`, `config_files/config.py`, `trackmania_rl/multiprocess/learner_process.py`

This section documents the three penalty terms added to the reward function to discourage behaviours that are physically possible in the simulator but impossible or unnatural for a human driver. The goal is not to make the agent slower — it is to constrain the **policy space** to solutions that a human could realistically execute, which is a prerequisite for using the agent as a driver assistance or coaching tool.

---

### Reward scale reference

Every design decision below is anchored to the base reward scale. At a typical racing speed of ~100 km/h:

| Component | Value per 50 ms step |
|---|---|
| Time penalty (`constant_reward_per_ms × 50`) | **−0.060** |
| Progress reward (~1.4 m at 100 km/h) | **+0.014** |
| Net reward per step at racing speed | **≈ −0.032 to −0.046** |

A penalty has to be in this ballpark to matter. Too small and the agent ignores it; too large and it destabilises learning by dominating the gradient signal.

The chosen coefficient for **all three penalties is −0.05**, which is approximately **1.0–1.5× the magnitude of one step's net reward**. This is the standard RL engineering rule of thumb for auxiliary penalties: large enough to shape behaviour, small enough not to destroy the primary signal.

---

### Penalty 1 — Steering oscillation

**Config key:** `humanlike_oscillation_penalty_schedule`  
**Default value:** `[(0, -0.05)]`

#### What it penalises
Rapid left↔right direction reversals. At 50 ms/step, a fully alternating L-R-L-R pattern produces a flip every step. Human minimum reaction time for a deliberate direction change is ~150–200 ms, making single-step reversals physiologically impossible.

#### Detection
A **prefix-sum sliding window** approach over the rollout:

1. A pre-pass scans every action and marks `flip_at_step[k] = 1` whenever action `k` had the **opposite** non-neutral steering direction from the previous non-neutral action (i.e. a direct L→R or R→L; neutral steps are ignored).
2. A prefix sum `flip_cumsum` is built so that the number of flips in any window `[i-W, i-1]` can be queried in O(1) as `flip_cumsum[i] - flip_cumsum[i-W]`.

#### Penalty formula
```
penalty(i) = humanlike_oscillation_penalty × max(0, flips_in_window − 1)
```

Window size `W = 4` steps = **200 ms**.

The `max(0, flips − 1)` shape gives **one free flip per 200 ms window** — a single direction change is a normal cornering correction and must not be penalised. Every flip beyond the first is an extra unit of penalty:

| Flips in 200 ms window | Multiplier | Effective penalty | Interpretation |
|---|---|---|---|
| 0 or 1 | 0 | 0.00 | Normal driving |
| 2 | 1 | −0.05 | L→R→L in ~150 ms — borderline |
| 3 | 2 | −0.10 | L→R→L→R in 200 ms — clearly inhuman |
| 4 | 3 | −0.15 | Maximum possible tapping rate |

At the maximum tapping frequency (4 flips, −0.15/step), the penalty is **3.3× the magnitude of the net progress reward**, producing a very strong gradient signal away from this behaviour.

#### Why −0.05 specifically
The physics gain from steering oscillation (exploiting the suspension/friction model) is empirically at most a 5–15% speed increase, translating to roughly +0.002 to +0.004 extra reward per step. At 2 flips, the penalty of −0.05 already outweighs this by 12–25×. The coefficient is conservative enough that **one natural cornering correction never triggers a penalty**, while still making rapid tapping clearly unprofitable.

---

### Penalty 2 — Brake tap (minimum hold duration)

**Config key:** `humanlike_brake_tap_penalty_schedule`  
**Default value:** `[(0, -0.05)]`

#### What it penalises
Brake presses held for fewer than **3 consecutive steps (150 ms)**. The minimum deliberate braking duration for a human is ~150–200 ms (one conscious press-and-hold cycle). Any braking event shorter than this is a micro-tap that the simulator responds to but no human could reproduce intentionally.

#### Detection
A forward-scan pre-pass:

1. Tracks `brake_hold` — a running counter of consecutive steps with brake active.
2. The moment brake turns off, if `0 < brake_hold < 3`, the release step is tagged in `brake_tap_penalty_at[idx]`.
3. In the main reward loop, `reward_into[idx] += brake_tap_penalty_at[idx]`.

This is an **event penalty** (one hit per tap occurrence), not a per-step penalty. Via the n-step return (n=3), it propagates backwards approximately 3 steps, so:

```
total effective signal ≈ −0.05 × (1 + γ + γ²) ≈ −0.15
```

which is comparable to the total reward of 3 normal steps. The agent strongly associates the entire short-brake sequence with a penalty, not just the release step.

#### Why 3 steps (150 ms) as the threshold
At 50 ms/step, the resolution is too coarse to distinguish a 10 ms tap from a 50 ms tap — both appear as 1 step. The threshold of 3 steps is chosen to match the lower bound of human deliberate braking (150 ms), accepting that 1–2 step braking (50–100 ms) is either an accident or an AI exploit. Raising this threshold (e.g. to 4 steps = 200 ms) would be more conservative; lowering it to 2 steps would allow 50 ms taps through unchecked.

#### Why −0.05 specifically
A short brake tap in TM can improve corner positioning by avoiding scrubbing. The benefit is at most ~0.005–0.02 reward per transition. The event penalty of −0.05 (which propagates to a total of ~−0.15 across 3 experiences) outweighs this by 7–30×, making micro-tapping clearly suboptimal.

---

### Penalty 3 — Neo slide at low speed

**Config key:** `humanlike_low_speed_slide_penalty_schedule`  
**Default value:** `[(0, -0.05)]`

#### What it penalises
Any wheel in a sliding state while forward speed is below **10 m/s (36 km/h)**. At high speed, wheel sliding is a legitimate technique (speedslide) already handled by `engineered_neoslide_reward` and `engineered_speedslide_reward`. At low speed, sliding is an AI physics exploit — it occurs when the agent spins in place, recovers from a wall hit awkwardly, or exploits the start-of-race physics. No human driver intentionally produces sustained low-speed wheel slides.

#### Detection
Checked directly in the main reward loop per step using two `state_float` indices:

| Index | Value |
|---|---|
| `state_float[21:25]` | `is_sliding` flag for each of the 4 wheels (float 0.0/1.0) |
| `state_float[58]` | Forward speed in car reference frame (m/s) |

```python
if any(is_sliding) and abs(speed_forward) < 10.0:
    reward_into[i] += humanlike_low_speed_slide_penalty
```

This is a **per-step penalty** — the longer the agent stays in this bad state, the more it accumulates. 10 steps of low-speed sliding (500 ms) = 10 × −0.05 = **−0.50**, which is roughly equivalent to losing 8 seconds of race time via the time penalty alone.

#### Why 10 m/s (36 km/h) as the threshold
This boundary cleanly separates two regimes:
- **Below 10 m/s:** sliding is almost always non-intentional (spin-outs, start chaos, wall bouncing).
- **Above 10 m/s:** sliding can be a deliberate speedslide. The existing `engineered_neoslide_reward` (lateral speed ≥ 2 m/s) and `engineered_speedslide_reward` (quality ≈ 1.0) already handle the high-speed sliding regime and should not be interfered with.

#### Why −0.05 specifically
At −0.05/step, low-speed sliding is penalised at **the same magnitude as one step's time penalty** (−0.06). Combined, the agent loses −0.11/step while sliding slowly — more than 2× the net reward magnitude at racing speed. The agent will strongly prefer escaping this state over staying in it. Choosing a value equal to the time penalty is principled: it says "sliding slowly is as bad as standing still", which is the correct relative assessment.

---

### Relationship to existing engineered rewards

| Reward / Penalty | Trigger | Sign | Scale | Interaction |
|---|---|---|---|---|
| `engineered_neoslide_reward` | Lateral speed ≥ 2 m/s | Configurable (+/−) | Configurable | Rewards high-speed sliding — does **not** conflict with neo slide penalty because the speed threshold (36 km/h) separates them cleanly |
| `engineered_speedslide_reward` | All 4 wheels grounded, quality ≈ 1.0 | Positive | Configurable | Rewards **perfect** high-speed slides — entirely orthogonal to the low-speed slide penalty |
| `humanlike_low_speed_slide_penalty` | Any sliding wheel + speed < 36 km/h | Negative | −0.05/step | Fills the gap left by the above two — penalises the bad low-speed regime they ignore |
| `engineered_kamikaze_reward` | Actions 0–2 OR ≤1 wheel grounded | Configurable | Configurable | Partially overlaps with low-speed slide (a spinning car often has few wheels grounded) but is conceptually distinct |
| `humanlike_oscillation_penalty` | 2+ L↔R flips in 200 ms | Negative | −0.05 × (flips−1)/step | No overlap with any existing reward — addresses input frequency which none of the original rewards touch |
| `humanlike_brake_tap_penalty` | Brake held < 150 ms | Negative | −0.05 per event | No overlap — addresses input duration which none of the original rewards touch |

---

### How to tune

All three are exposed as linear schedules. The schedule format is `[(step, value), ...]` with linear interpolation between setpoints, keyed on `cumul_number_frames_played`.

**Enable immediately (current default):**
```python
humanlike_oscillation_penalty_schedule    = [(0, -0.05)]
humanlike_brake_tap_penalty_schedule      = [(0, -0.05)]
humanlike_low_speed_slide_penalty_schedule = [(0, -0.05)]
```

**Ramp in after initial convergence (recommended for training from scratch):**
```python
humanlike_oscillation_penalty_schedule    = [(0, 0), (500_000, -0.05)]
humanlike_brake_tap_penalty_schedule      = [(0, 0), (500_000, -0.05)]
humanlike_low_speed_slide_penalty_schedule = [(0, 0), (500_000, -0.05)]
```

**Strengthen if behaviour persists after many training steps:**
```python
humanlike_oscillation_penalty_schedule    = [(0, 0), (500_000, -0.05), (3_000_000, -0.10)]
```

**Disable a penalty entirely:**
```python
humanlike_oscillation_penalty_schedule    = [(0, 0)]
```

To apply changes during a running training session, edit `config_copy.py` directly — changes take effect on the next rollout without restarting or clearing the replay buffer.

---

---

## 24. Braking Aggression Conditioning

**Files:** `config_files/config.py`, `trackmania_rl/tmi_interaction/game_instance_manager.py`, `trackmania_rl/buffer_management.py`, `trackmania_rl/multiprocess/learner_process.py`, `config_files/state_normalization.py`

This section documents the braking aggression system, which conditions the agent on a target driver profile and aligns the agent's brake usage distribution to that target using a principled proper scoring rule.

---

### Motivation

The Racer Helper project generates a **personalized ghost** that mimics a specific driver's style — not a robot-optimal line. One of the most driver-distinctive behaviours is braking aggression: aggressive drivers brake late and hard at every corner; smooth drivers coast and brake gently. The agent needs to:

1. **Receive** the target aggression as a conditioning input, so a single trained model can produce different driving styles.
2. **Be shaped** by a reward signal that pushes its empirical brake usage rate toward the target, without destroying the primary laptime objective.

---

### Part 1 — State Conditioning (UVFA)

**Mechanism:** `braking_aggression` is appended to the float feature vector as the last element (index 184). It is a scalar in `[0, 1]`:

| Value | Interpretation |
|---|---|
| 0.0 | Never brakes — pure coasting driver |
| 0.3 | Brakes sparingly, mostly on fast corners (default) |
| 0.5 | Moderate — brakes at most corners |
| 1.0 | Brakes maximally at every braking opportunity |

**Why concatenation works:** This is the **Universal Value Function Approximator (UVFA)** pattern (Schaul et al., 2015). The network learns to output different Q-value distributions for the same physical state depending on the conditioning variable. When `braking_aggression = 0.3`, the network avoids brake actions in situations where they would incur the Brier-score penalty; when `braking_aggression = 1.0`, it favours brake actions in those same situations.

**Implementation:**

```python
# game_instance_manager.py — appended at construction time each step
floats = np.hstack([
    0,                                  # index 0  (overwritten by mini-race timer at collation)
    previous_actions_features,          # indices 1–20
    car_gear_and_wheels,                # indices 21–52
    angular_velocity,                   # indices 53–55
    linear_velocity,                    # indices 56–58
    y_map_vector,                       # indices 59–61
    zone_centers_in_car_frame,          # indices 62–181
    margin_to_finish,                   # index 182
    is_freewheeling,                    # index 183
    config_copy.braking_aggression,     # index 184  ← NEW
])
```

**Normalization** (`state_normalization.py`):
- Mean: `0.5` (centre of `[0, 1]`)
- Std: `0.3` (covers the realistic driver range with ≈ 1.7σ from each extreme)

After normalization: `(braking_aggression − 0.5) / 0.3`, so a value of `0.5` maps to `0.0` (neutral), `0.0` maps to `−1.67`, and `1.0` maps to `+1.67`.

---

### Part 2 — Brier-Score Reward Penalty

#### What is a proper scoring rule?

A **proper scoring rule** for a binary event is a loss function `L(p, outcome)` such that its expected value (over the true outcome distribution) is minimised if and only if the reported probability `p` equals the true event probability. In other words, the agent is incentivised to report — or in our case, *produce* — the correct probability.

The **Brier score** is the canonical proper scoring rule for binary events:

```
L_Brier(p, y) = (y − p)²
```

where `y ∈ {0, 1}` is the observed outcome and `p ∈ [0, 1]` is the predicted probability.

Expected Brier score when the agent brakes with probability `f`:

```
E[L_Brier(α, brake)] = f × (1 − α)² + (1 − f) × (0 − α)²
                     = f × (1 − α)² + (1 − f) × α²
```

Taking the derivative and setting to zero:

```
d/df = (1 − α)² − α² = 0
     → f = α    (i.e., calibrated probability = target)
```

This confirms the proper scoring property: **the expected penalty is minimised exactly when `P(brake|state) = braking_aggression`**.

#### Per-step reward formula

At each non-terminal step `i`:

```
r_brake(i) = coeff × (brake(action_i) − braking_aggression)²
```

| Scenario | Penalty |
|---|---|
| `braking_aggression = 1.0`, agent brakes | `coeff × (1−1)² = 0` |
| `braking_aggression = 1.0`, agent does not brake | `coeff × (0−1)² = coeff` |
| `braking_aggression = 0.5`, agent brakes | `coeff × (1−0.5)² = 0.25 × coeff` |
| `braking_aggression = 0.5`, agent does not brake | `coeff × (0−0.5)² = 0.25 × coeff` |
| `braking_aggression = 0.0`, agent brakes | `coeff × (1−0)² = coeff` |
| `braking_aggression = 0.0`, agent does not brake | `coeff × (0−0)² = 0` |

With `coeff = −0.05`, the maximum penalty per step is `−0.05`, consistent with the other human-likeness coefficients.

#### Connection to the IQN loss

The braking aggression penalty is incorporated into the reward, not the loss function form. The IQN quantile Huber loss remains:

```
L_IQN = E_{τ,τ'}[ρ_τ(r_total + γ × Z_{τ'}(s', a*) − Z_τ(s, a))]
```

where `r_total` now includes `r_brake`. The distributional Bellman targets implicitly encode the braking-alignment objective across the full return distribution. High-aggression drivers get targets that assign higher Q-values to brake actions; low-aggression drivers get targets that penalise unnecessary braking. The network learns both simultaneously via the conditioning input.

---

### Configuration

**`config.py` keys:**

```python
braking_aggression = 0.3
# Float in [0, 1]. Set this to match the target driver's characteristic
# brake usage rate. 0.3 is a reasonable default for a smooth but
# competent driver who brakes at fast corners only.

humanlike_braking_aggression_reward_schedule = [(0, -0.05)]
# Schedule format: [(step, coefficient), ...] — linear interpolation.
# Coefficient is negative (penalty). Maximum |coeff| = 0.05 is recommended
# (matches other human-likeness penalties).
```

**Ramp-in schedule (recommended when training from scratch):**
```python
humanlike_braking_aggression_reward_schedule = [(0, 0), (500_000, -0.05)]
```

**Disable the penalty entirely (state conditioning remains active):**
```python
humanlike_braking_aggression_reward_schedule = [(0, 0)]
```

**Change driver profile at mid-run (edit `config_copy.py`):**
```python
braking_aggression = 0.7  # switch to a more aggressive driver profile
```

---

### Relationship to existing penalties

| Penalty | What it targets | Interaction with braking aggression |
|---|---|---|
| `humanlike_brake_tap_penalty` | Brake presses shorter than 150 ms | Orthogonal — tap detection enforces minimum hold duration; aggression shapes *frequency*, not *duration* |
| `humanlike_low_speed_slide_penalty` | Sliding at speed < 36 km/h | No overlap — addresses sliding, not brake usage |
| `humanlike_oscillation_penalty` | Steering L↔R flips | No overlap — addresses steering, not braking |
| `engineered_speedslide_reward` | High-speed lateral sliding quality | No conflict — neither triggers at the same condition |

---

### Design notes

- **Why Brier score rather than cross-entropy?** The Brier score is bounded (`[0, 1]`) and has bounded gradients, making it safer to incorporate into a reward without destabilising IQN training. Cross-entropy loss diverges as the predicted probability approaches 0 or 1, which would produce unbounded rewards.
- **Why per-step rather than episode-level?** Episode-level reward (e.g. penalise total brake count deviation) provides a much weaker and more delayed signal. Per-step rewards propagate cleanly through the n-step return and allow the network to learn which *specific states* should trigger braking.
- **Why does the conditioning input (index 184) survive collation?** `buffer_utilities.py` only overwrites index 0 (the mini-race timer). All other indices, including 184, pass through unchanged from the stored `state_float`. The network therefore always sees the correct `braking_aggression` value that was active during the rollout that generated that experience.

---

*Generated from source — covers every module, class, function, and design decision in the Linesight codebase.*
