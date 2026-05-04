# Racer Helper — Human-Achievable Personalized Ghost for Trackmania

## Overview
**Racer Helper** gives competitive drivers a **personalized, realistic target** instead of an unrealistic “perfect lap.”  
It helps drivers improve faster by showing the **best line they can actually execute**, not the best line a robot could execute.

The system is designed for **Trackmania** and produces a driver-specific “best version” ghost plus actionable feedback.
It is a deep reinforcement learning system that trains an AI agent to drive Trackmania Nations Forever at superhuman speeds, then generates a personalized ghost replay that mirrors a specific human driver's style. We use linesight as our base model , and we fine tune it to the target driver's profile. The agent uses an Implicit Quantile Network (IQN) to learn the full distribution of expected returns, while a set of conditioning variables — braking aggression, risk tolerance, and human-likeness penalties — constrain and shape its behavior to match a target driver profile rather than a robotic optimum.
---

## Repository Layout

```
RacerHelper-DriverConditionedRL/
├── linesight/                  # RL engine (forked from linesight)
│   ├── config_files/           # Training config — edit config.py to set driver profile
│   ├── trackmania_rl/          # RL core: agents, buffer, reward shaping, TMI interface
│   ├── scripts/
│   │   ├── train.py            # Training entry point
│   │   └── extract_driver_profile.py  # Parse a replay → driver config values
│   ├── custom_maps/            # Track .Gbx files
│   └── maps/                   # Preprocessed .npy map data
├── services/
│   ├── service1/               # Frontend + backend
│   ├── service2/               # Driver profile API (FastAPI)
│   └── service3/               # Training orchestration + TMNF Docker scripts
├── models/                     # Saved weights per track (Bahrain, Hockolicious, …)
├── replays/                    # Driver .Replay.Gbx files used for profiling
├── setup_scripts/              # One-time setup for linesight and TMNF
└── docker-compose.yml
```

---

## 1) Problem Definition

| Document | Description |
|---|---|
| [Linesight Technical Reference](./docs/linesight-technical.md) | Deep-dive into the base RL system: IQN algorithm, neural network architecture, mini-race logic, game interaction layer, hyperparameters (sections 1–22) |
| [RacerHelper Extensions](./docs/racerhelper-extensions.md) | What we added: human-likeness penalties, braking aggression conditioning, risk tolerance conditioning — with full reward design rationale (sections 23–25) |
| [run_services.md](./run_services.md) | Step-by-step guide for running the services locally or via Docker |

---

# Running the Services

## Architecture

```
Browser
  │
  └─► Service 1  :8001  (UI + API gateway)
        │
        └─► Service 2  :8002  (driver profile extraction)
        │
        └─► Service 3  :8003  (replay generation — not yet implemented)
```

Service 1 is the only public-facing service. The browser never talks to Service 2 or 3 directly — Service 1 proxies everything.

---

## Assumptions

- **Docker Desktop** is installed and running (for the Docker path).
- **Python 3.11+** is available (for the local path).
- The `linesight/` folder is present at the repo root — Service 2 copies `linesight/scripts/extract_driver_profile.py` into its container at build time.
- `pygbx==0.3`, `numpy`, and `scipy` are installable in the Service 2 environment. `pygbx` is the Trackmania replay parser; without it Service 2 will start but every upload will fail.

---

## Option A — Docker Compose (recommended)

```powershell
# 1. From the repo root
cd RacerHelper-DriverConditionedRL

# 2. Edit .env if needed (defaults are fine for local Docker)
#    See "Environment variables" section below.

# 3. Build and start both services
docker compose up --build

# 4. Open the UI
start http://localhost:8001
```

To stop:
```powershell
docker compose down
```

To rebuild a single service after a code change:
```powershell
docker compose up --build service2
```

---

## Option B — Local (no Docker)

Run each service in its own terminal. Service 2 must start first.

### Terminal 1 — Service 2

```powershell
cd services\service2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8002 --reload
```

### Terminal 2 — Service 1

```powershell
cd services\service1\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8001 --reload
```

Then open `http://localhost:8001`.

> **Important:** When running locally, `SERVICE2_URL` must point to `localhost`, not the Docker service name. Either set it in your shell or edit `.env`:
> ```
> SERVICE2_URL=http://localhost:8002
> SERVICE3_URL=http://localhost:8003
> ```

---

## Environment variables

All variables live in `.env` at the repo root. Docker Compose reads this file automatically.

| Variable | Default (Docker) | Description |
|---|---|---|
| `SERVICE1_PORT` | `8001` | Host port for Service 1 |
| `SERVICE2_PORT` | `8002` | Host port for Service 2 |
| `SERVICE3_PORT` | `8003` | Host port for Service 3 (reserved) |
| `SERVICE2_URL` | `http://service2:8002` | Where Service 1 calls Service 2. Use `http://localhost:8002` for local dev. |
| `SERVICE3_URL` | `http://service3:8003` | Where Service 1 will call Service 3. Use `http://localhost:8003` for local dev. |
| `LINESIGHT_SCRIPTS_PATH` | `/app/scripts` | Path to `extract_driver_profile.py` inside the Service 2 container. Do not change unless you move the file. |
| `SERVICE1_CORS_ORIGINS` | `*` | Comma-separated allowed origins for Service 1. Lock down to your domain in production. |
| `SERVICE2_CORS_ORIGINS` | `http://localhost:8001` | Comma-separated allowed origins for Service 2. Browsers never call Service 2 directly, so this is defence-in-depth only. |

---

## Adding Service 3

When Service 3 is ready:

1. Create `services/service3/Dockerfile` and its code.
2. Uncomment the `service3` block in `docker-compose.yml`.
3. Service 1 already reads `SERVICE3_URL` from the environment — no code changes needed there.
4. Implement the actual HTTP call in `services/service1/backend/main.py` inside the `generate` endpoint (currently a stub).



## Team
- Jad Al Lakkis  
- Ibrahim Khaled