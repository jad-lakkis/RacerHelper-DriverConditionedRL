# Racer Helper — Human-Achievable Personalized Ghost for Trackmania

## Overview

**Racer Helper** gives competitive drivers a **personalized, realistic target** instead of an unrealistic “perfect lap.”  
It helps drivers improve faster by showing the **best line they can actually execute**, not the best line a robot could execute.

The system is designed for **Trackmania** and produces a driver-specific “best version” ghost plus actionable feedback.
It is a deep reinforcement learning system that trains an AI agent to drive Trackmania Nations Forever at superhuman speeds, then generates a personalized ghost replay that mirrors a specific human driver's style. We use linesight as our base model , and we fine tune it to the target driver's profile. The agent uses an Implicit Quantile Network (IQN) to learn the full distribution of expected returns, while a set of conditioning variables — braking aggression, risk tolerance, and human-likeness penalties — constrain and shape its behavior to match a target driver profile rather than a robotic optimum.



# Running the Services

## Architecture

```
Browser
  │
  └─► Service 1  :8001  (UI + API gateway)
        │
        └─► Service 2  :8002  (driver profile extraction)
        │
        └─► Service 3  <PUBLIC_IP>:<PUBLIC_PORT> (training orchestration + TMNF Docker In vast.ai setup more details are in )  
```

Service 1 is the only public-facing service. The browser never talks to Service 2 or 3 directly — Service 1 proxies everything.


## 1) Setup of Vast.ai for Service 3
SERVICE3 SETUP MUST BE RUN FIRST BECAUSE it requires setting up SERVICE3_URL=IP_ADDRESS:PORT in the .env file for service1 and service2 to connect to it. 

So Prelimnary steps before 1 and 2 is to run serrvice 3 

Follow this guide to setup service3 on vast.ai : [Setup on Vast.ai](./docs/setup_vast_ai/setup_on_vast.ai.md)

---

## 2) Docker Compose - Services 1 & 2 


```powershell
# 1. From the repo root
cd RacerHelper-DriverConditionedRL

# 2. Make sure the .env file is properly set up with SERVICE3_URL pointing to your service3 instance  on vast.ai

# 3. Build and start both services
docker compose up --build

```
Then open `http://localhost:8001`.




#  Technical Documentation

| Document | Description |
|---|---|
| [Linesight Technical Reference](./docs/linesight-technical.md) | Deep-dive into the base RL system: IQN algorithm, neural network architecture, mini-race logic, game interaction layer, hyperparameters (sections 1–22) |
| [RacerHelper Extensions](./docs/racerhelper-extensions.md) | What we added: human-likeness penalties, braking aggression conditioning, risk tolerance conditioning — with full reward design rationale (sections 23–25) |
---

## Team
- Jad Al Lakkis  
- Ibrahim Khaled

