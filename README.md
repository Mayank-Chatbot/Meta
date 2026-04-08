---
title: "MetaMind"
emoji: "🤖"
colorFrom: "blue"
colorTo: "purple"
sdk: docker
sdk_version: "latest"
python_version: "3.11"
app_file: Dockerfile
pinned: false
---

# MetaMind — Self-Improving AI Environment

> An OpenEnv-compliant environment where AI agents improve their problem-solving strategy through iterative attempts using Contextual Bandit → Multi-Armed Bandit → Q-Learning.

---

## What It Does

MetaMind presents a task to three progressively sophisticated RL algorithms. Each attempt is quality-graded (0.0–1.0). If the score is below threshold, the system escalates to the next algorithm — adapting strategy until quality is met or all levels are exhausted.

```
User Task
    │
    ▼
┌─────────────────────────────┐
│  Level 1: Contextual Bandit │  threshold: 70%   reward: +1.0 / -0.5
│  Greedy exploit, no explore  │
└─────────────┬───────────────┘
              │ FAIL
              ▼
┌─────────────────────────────┐
│  Level 2: Multi-Armed Bandit│  threshold: 83%   reward: +1.5 / -0.3
│  UCB1 — 5 strategy arms      │
└─────────────┬───────────────┘
              │ FAIL
              ▼
┌─────────────────────────────┐
│  Level 3: Q-Learning        │  threshold: 95%   reward: +2.0 / -0.1
│  Full RL with Q(s,a) table   │
└─────────────────────────────┘
```

---

## Project Structure

```
metamind/
├── meta_ai_env.py    ← FastAPI backend (OpenEnv-compliant)
├── index.html        ← Frontend web UI
├── requirements.txt  ← Python dependencies
├── .env.example      ← API key template
├── Dockerfile        ← Container deployment
├── openenv.yaml      ← OpenEnv specification
└── README.md
```

---

## Quick Start (VS Code)

### 1. Install prerequisites
```bash
python --version   # needs 3.10+
node --version     # not required, just for reference
```

### 2. Create project folder
```bash
mkdir metamind && cd metamind
# Copy all project files here
code .
```

### 3. Create virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set your API key
```bash
# Copy .env.example to .env
cp .env.example .env


### 6. Start the backend
```bash
uvicorn meta_ai_env:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 7. Open the frontend
Open `index.html` directly in your browser (double-click or drag into browser).

Type a task and click **Run Pipeline →**

---

## API Reference

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/`       | API info                             |
| POST   | `/reset`  | Reset env with new task              |
| POST   | `/step`   | Execute one algorithm step           |
| GET    | `/state`  | Current environment state            |
| POST   | `/run`    | Run full pipeline (all 3 algorithms) |
| GET    | `/agents` | Q-table, arm stats, CB values        |

> Note: `POST /reset` accepts an empty JSON body and defaults to `"initialize task"` to satisfy OpenEnv validator expectations.

### POST /run (recommended)
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Explain how neural networks learn"}'
```

### Response shape
```json
{
  "run_id": "a3f8c2e1",
  "task": "Explain how neural networks learn",
  "steps": [...],
  "total_reward": 1.42,
  "winning_algorithm": "Multi-Armed Bandit",
  "final_score": 0.84,
  "adaptation_gain": 0.21,
  "attempts_used": 2,
  "success": true,
  "duration_seconds": 8.3
}
```

---

## Reward Function

```
Total Reward = Base Reward − Step Penalty + Adaptation Bonus

Base Reward:
  PASS → +1.0 (CB) / +1.5 (MAB) / +2.0 (QL)
  FAIL → -0.5 (CB) / -0.3 (MAB) / -0.1 (QL)

Step Penalty:
  attempt_number × per_algo_penalty (0.05 / 0.03 / 0.01)

Adaptation Bonus:
  0.7 × (current_score − previous_score)  [only if improved]
```

---

## Algorithms

### Level 1 — Contextual Bandit
Pure greedy exploitation. Maps context (task length) to action values using exponential moving average. Updates after each attempt. No exploration.

### Level 2 — Multi-Armed Bandit (UCB1)
5 strategy arms: `direct_answer`, `step_by_step_reasoning`, `analogy_and_example`, `structured_breakdown`, `first_principles`. Selects arm using UCB1 formula:
```
arm = argmax[ Q(a) + √2 · √(ln(t) / N(a)) ]
```
Updates arm values using incremental mean. Agents persist across runs, learning over time.

### Level 3 — Q-Learning
State = (attempt_number, score_bucket). 5 actions. Updates using Bellman equation:
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') − Q(s,a)]
α=0.15, γ=0.90, ε=0.10
```
Q-table persists across runs — the agent genuinely learns over multiple sessions.

---

## Docker Deployment

```bash
# Build
docker build -t metamind .

# Run (pass your API key)
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... metamind
```

Then open `index.html` in your browser.

---

## OpenEnv Compliance

- [x] Typed `Observation`, `Action`, `Reward` models (Pydantic)
- [x] `step()`, `reset()`, `state()` API endpoints
- [x] Minimum 3 task difficulties (easy/medium/hard via thresholds)
- [x] Deterministic quality grader (LLM evaluator, 0.0–1.0)
- [x] Dense reward signals with partial progress
- [x] Baseline inference via `POST /run`
- [x] `openenv.yaml` specification
- [x] Dockerfile for reproducible deployment

---

## 30-Second Pitch

> "Current AI systems solve problems but don't improve *how* they solve them.
>
> MetaMind is an OpenEnv environment where three RL algorithms — Contextual Bandit, Multi-Armed Bandit, and Q-Learning — attempt the same task in sequence, each learning from the previous failure.
>
> The system scores not just on correctness but on efficiency and adaptation — and rewards strategy improvement across attempts.
>
> This shifts AI evaluation from static problem-solving to dynamic self-improvement."
"# Meta" 
