"""
╔══════════════════════════════════════════════════════════════════╗
║          MetaMind — OpenEnv Self-Improvement Environment         ║
║  Contextual Bandit → Multi-Armed Bandit → Q-Learning             ║
╚══════════════════════════════════════════════════════════════════╝

Setup:
  1. pip install -r requirements.txt
  2. Copy .env.example → .env and add your ANTHROPIC_API_KEY
  3. uvicorn meta_ai_env:app --reload --port 8000
  4. Open index.html in your browser
"""

from __future__ import annotations
import os, json, math, random, time, uuid
from typing import Literal, Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MetaMind OpenEnv",
    description="Self-Improving AI Environment with Contextual Bandit, MAB, and Q-Learning",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))

# ─────────────────────────────────────────────────────────────────
# OpenEnv Typed Models
# ─────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task: str
    attempt_number: int
    algorithm: str
    level: str
    previous_score: Optional[float] = None
    previous_response: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

class Action(BaseModel):
    type: Literal["reason", "calculate", "retry", "change_strategy", "submit"]
    content: str
    strategy: Optional[str] = None
    arm: Optional[str] = None
    q_action: Optional[str] = None

class Reward(BaseModel):
    base: float
    adaptation_bonus: float
    step_penalty: float
    total: float
    passed: bool
    score: float
    threshold: float
    reason: str

class StepResult(BaseModel):
    run_id: str
    step: int
    observation: Observation
    action: Action
    reward: Reward
    response: str
    algorithm: str
    level: str
    algo_key: str
    done: bool
    timestamp: float

class RunResult(BaseModel):
    run_id: str
    task: str
    steps: List[StepResult]
    total_reward: float
    winning_algorithm: Optional[str]
    winning_algo_key: Optional[str]
    final_score: float
    adaptation_gain: float
    attempts_used: int
    success: bool
    duration_seconds: float

# ─────────────────────────────────────────────────────────────────
# Contextual Bandit — Level 1
# ─────────────────────────────────────────────────────────────────

class ContextualBandit:
    """
    Greedy exploitation: always picks the action with highest estimated
    reward for the current context. No exploration — pure greedy.
    Context is encoded as a simple feature vector (task length bucket).
    """
    def __init__(self):
        self.context_values: Dict[str, float] = {}

    def get_context_key(self, task: str) -> str:
        length = len(task)
        if length < 50:   return "short"
        if length < 150:  return "medium"
        return "long"

    def select_action(self, task: str) -> str:
        ctx = self.get_context_key(task)
        key = f"{ctx}_direct"
        return key if self.context_values.get(key, 0) > 0 else "direct_answer"

    def update(self, task: str, action: str, reward: float):
        ctx = self.get_context_key(task)
        key = f"{ctx}_{action}"
        old = self.context_values.get(key, 0.0)
        self.context_values[key] = old * 0.9 + reward * 0.1  # exponential moving avg

    def get_system_prompt(self) -> str:
        return (
            "You are a Contextual Bandit AI operating at Level 1 (Basic). "
            "Your policy is pure greedy exploitation: give the single most direct, "
            "obvious answer based on this exact context. "
            "Be brief — 2 to 3 sentences only. No elaboration, no examples, no tangents. "
            "This simulates a zero-exploration greedy bandit policy."
        )

# ─────────────────────────────────────────────────────────────────
# Multi-Armed Bandit — Level 2 (UCB1)
# ─────────────────────────────────────────────────────────────────

class MultiArmedBandit:
    """
    Upper Confidence Bound (UCB1) exploration:
    selects arm = argmax[ Q(a) + C * sqrt(ln(t) / N(a)) ]
    Balances exploitation of known good arms with exploration of uncertain ones.
    """
    ARMS = [
        "direct_answer",
        "step_by_step_reasoning",
        "analogy_and_example",
        "structured_breakdown",
        "first_principles",
    ]
    C = math.sqrt(2)  # exploration constant

    def __init__(self):
        self.counts:  Dict[str, int]   = {a: 0   for a in self.ARMS}
        self.values:  Dict[str, float] = {a: 0.0 for a in self.ARMS}
        self.total:   int = 0

    def select_arm(self) -> str:
        self.total += 1
        # Always try unvisited arms first
        for arm in self.ARMS:
            if self.counts[arm] == 0:
                return arm
        # UCB1
        ucb_scores = {
            a: self.values[a] + self.C * math.sqrt(math.log(self.total) / self.counts[a])
            for a in self.ARMS
        }
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n  # incremental mean

    def get_arm_stats(self) -> Dict[str, dict]:
        return {
            a: {"count": self.counts[a], "value": round(self.values[a], 4)}
            for a in self.ARMS
        }

    def get_system_prompt(self, arm: str) -> str:
        arm_instructions = {
            "direct_answer":          "Give the most direct, concise answer possible.",
            "step_by_step_reasoning": "Break your answer into clear numbered steps.",
            "analogy_and_example":    "Use a concrete analogy or real-world example to explain.",
            "structured_breakdown":   "Use headings and bullet points for a structured response.",
            "first_principles":       "Reason from first principles — start from fundamentals.",
        }
        instruction = arm_instructions.get(arm, "Provide a thorough answer.")
        return (
            f"You are a Multi-Armed Bandit AI operating at Level 2 (Medium). "
            f"You use UCB1 exploration. The previous basic attempt was insufficient. "
            f"Selected arm strategy: '{arm.replace('_', ' ')}'. {instruction} "
            f"Provide a more thorough answer than the previous attempt (4–6 sentences). "
            f"Briefly mention which strategy arm you chose at the start."
        )

# ─────────────────────────────────────────────────────────────────
# Q-Learning Agent — Level 3
# ─────────────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
    State  = (attempt_number, score_bucket)
    Action = response strategy
    Learns optimal action selection across multiple runs.
    """
    ACTIONS = [
        "comprehensive_explanation",
        "structured_with_examples",
        "technical_deep_dive",
        "concise_and_precise",
        "socratic_reasoning",
    ]

    def __init__(self, alpha=0.15, gamma=0.9, epsilon=0.1):
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.alpha   = alpha    # learning rate
        self.gamma   = gamma    # discount factor
        self.epsilon = epsilon  # exploration rate

    def _state_key(self, attempt: int, prev_score: Optional[float]) -> str:
        bucket = "none" if prev_score is None else str(int(prev_score * 10))
        return f"attempt_{attempt}_score_{bucket}"

    def get_q(self, state: str, action: str) -> float:
        return self.q_table.get(state, {}).get(action, 0.0)

    def select_action(self, attempt: int, prev_score: Optional[float]) -> str:
        state = self._state_key(attempt, prev_score)
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)  # explore
        return max(self.ACTIONS, key=lambda a: self.get_q(state, a))  # exploit

    def update(self, attempt: int, prev_score: Optional[float],
               action: str, reward: float, next_score: float):
        state      = self._state_key(attempt, prev_score)
        next_state = self._state_key(attempt + 1, next_score)
        if state not in self.q_table:
            self.q_table[state] = {}
        best_next = max(self.get_q(next_state, a) for a in self.ACTIONS)
        old = self.get_q(state, action)
        self.q_table[state][action] = old + self.alpha * (
            reward + self.gamma * best_next - old
        )

    def get_q_table_summary(self) -> Dict[str, Any]:
        return {
            state: {a: round(v, 4) for a, v in actions.items()}
            for state, actions in self.q_table.items()
        }

    def get_system_prompt(self, action: str, prev_attempts: List[dict]) -> str:
        action_instructions = {
            "comprehensive_explanation": "Provide a comprehensive, thorough explanation covering all important aspects.",
            "structured_with_examples":  "Use clear structure with concrete examples for each key point.",
            "technical_deep_dive":       "Go deep technically — precise terminology, mechanisms, and nuance.",
            "concise_and_precise":       "Be maximally precise and complete but without unnecessary words.",
            "socratic_reasoning":        "Build understanding progressively, from simple to complex.",
        }
        instruction = action_instructions.get(action, "Provide the optimal answer.")
        prev_summary = ""
        if prev_attempts:
            lines = [
                f"  - {a['algorithm']} scored {a['score']*100:.0f}%: \"{a['response'][:100]}...\""
                for a in prev_attempts
            ]
            prev_summary = f"\n\nPrevious failed attempts:\n" + "\n".join(lines) + "\n\nLearn from these failures."
        return (
            f"You are a Q-Learning AI operating at Level 3 (Advanced). "
            f"You maintain a full state-action-reward table. "
            f"Q-table selected action: '{action.replace('_', ' ')}'. {instruction} "
            f"Produce your absolute best answer — comprehensive, accurate, well-organized. "
            f"No length restriction. Quality is the only objective.{prev_summary}"
        )

# ─────────────────────────────────────────────────────────────────
# Algorithm Configuration
# ─────────────────────────────────────────────────────────────────

ALGORITHMS = {
    "contextual_bandit": {
        "name": "Contextual Bandit",
        "level": "Basic",
        "threshold": 0.70,
        "reward_pass": 1.0,
        "reward_fail": -0.5,
        "step_penalty_per_attempt": 0.05,
        "color": "#6c63ff",
    },
    "multi_armed_bandit": {
        "name": "Multi-Armed Bandit",
        "level": "Medium",
        "threshold": 0.83,
        "reward_pass": 1.5,
        "reward_fail": -0.3,
        "step_penalty_per_attempt": 0.03,
        "color": "#22d3a5",
    },
    "q_learning": {
        "name": "Q-Learning",
        "level": "Advanced",
        "threshold": 0.95,
        "reward_pass": 2.0,
        "reward_fail": -0.1,
        "step_penalty_per_attempt": 0.01,
        "color": "#fbbf24",
    },
}
ALGO_ORDER = ["contextual_bandit", "multi_armed_bandit", "q_learning"]

# ─────────────────────────────────────────────────────────────────
# Singleton Agents (persist across requests, learn over time)
# ─────────────────────────────────────────────────────────────────

cb_agent  = ContextualBandit()
mab_agent = MultiArmedBandit()
ql_agent  = QLearningAgent()

# ─────────────────────────────────────────────────────────────────
# LLM Helpers
# ─────────────────────────────────────────────────────────────────

def llm(system: str, user: str, max_tokens: int = 1200) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


def evaluate(response: str, task: str, level: str) -> Dict[str, Any]:
    prompt = (
        f"Evaluate this AI response quality on a scale from 0.00 to 1.00.\n\n"
        f"Task: \"{task}\"\n"
        f"Algorithm level: {level}\n"
        f"Response: \"{response[:700]}\"\n\n"
        f"Scoring:\n"
        f"  0.00–0.40 → Wrong, irrelevant, or dangerously vague\n"
        f"  0.40–0.60 → Partially correct, missing key points\n"
        f"  0.60–0.75 → Correct but too brief or incomplete\n"
        f"  0.75–0.88 → Good — correct and reasonably thorough\n"
        f"  0.88–1.00 → Excellent — comprehensive, accurate, well-structured\n\n"
        f"Reply ONLY with valid JSON (no markdown, no backticks):\n"
        f"{{\"score\": 0.XX, \"reason\": \"one concise sentence\"}}"
    )
    raw = llm(
        system="You are a strict AI quality evaluator. Output only valid JSON.",
        user=prompt,
        max_tokens=120,
    )
    try:
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
        return {"score": float(data["score"]), "reason": str(data.get("reason", ""))}
    except Exception:
        import re
        m = re.search(r'"score"\s*:\s*([\d.]+)', raw)
        score = float(m.group(1)) if m else 0.5
        return {"score": min(1.0, max(0.0, score)), "reason": "auto-parsed"}

# ─────────────────────────────────────────────────────────────────
# Environment State
# ─────────────────────────────────────────────────────────────────

class Environment:
    def __init__(self):
        self.task = ""
        self.run_id = ""
        self.attempts: List[Dict] = []
        self.algo_idx = 0
        self.done = False
        self.start_time = 0.0

    def reset(self, task: str):
        self.task = task
        self.run_id = str(uuid.uuid4())[:8]
        self.attempts = []
        self.algo_idx = 0
        self.done = False
        self.start_time = time.time()

    def state_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "current_algo": ALGO_ORDER[self.algo_idx] if self.algo_idx < 3 else "complete",
            "attempts_count": len(self.attempts),
            "done": self.done,
        }

_env = Environment()

# ─────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str

class RunRequest(BaseModel):
    task: str

# ─────────────────────────────────────────────────────────────────
# OpenEnv API Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "MetaMind OpenEnv",
        "version": "1.0.0",
        "description": "Self-improving AI environment: CB → MAB → QL",
        "endpoints": {
            "POST /run":   "Run full pipeline (recommended)",
            "POST /reset": "Reset environment with a new task",
            "POST /step":  "Execute one algorithm step",
            "GET  /state": "Get current environment state",
            "GET  /agents":"Get agent internal states (Q-table, arm stats)",
        },
    }


class ResetResponse(BaseModel):
    status: str
    run_id: str
    task: str
    observation: Observation


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    _env.reset(req.task)
    initial_obs = Observation(
        task=_env.task,
        attempt_number=1,
        algorithm=ALGORITHMS[ALGO_ORDER[0]]["name"],
        level=ALGORITHMS[ALGO_ORDER[0]]["level"],
        previous_score=None,
        previous_response=None,
        context={},
    )
    return ResetResponse(
        status="ok",
        run_id=_env.run_id,
        task=_env.task,
        observation=initial_obs,
    )


@app.get("/state")
def get_state():
    return _env.state_dict()


@app.get("/agents")
def get_agents():
    return {
        "contextual_bandit": {
            "context_values": cb_agent.context_values,
        },
        "multi_armed_bandit": {
            "arm_stats": mab_agent.get_arm_stats(),
            "total_pulls": mab_agent.total,
        },
        "q_learning": {
            "q_table": ql_agent.get_q_table_summary(),
            "alpha": ql_agent.alpha,
            "gamma": ql_agent.gamma,
            "epsilon": ql_agent.epsilon,
        },
    }


@app.post("/step", response_model=StepResult)
def step():
    if _env.done or _env.algo_idx >= 3:
        raise HTTPException(400, "Environment done. Call POST /reset first.")

    algo_key = ALGO_ORDER[_env.algo_idx]
    algo     = ALGORITHMS[algo_key]
    attempt  = _env.algo_idx + 1
    prev     = _env.attempts[-1] if _env.attempts else None

    # ── Select action & build prompt ──────────────────────────
    action_type = "reason"
    arm, q_action = None, None

    if algo_key == "contextual_bandit":
        action_name = cb_agent.select_action(_env.task)
        system_p    = cb_agent.get_system_prompt()
        user_p      = _env.task
        action_type = "calculate" if any(c.isdigit() for c in _env.task) else "reason"

    elif algo_key == "multi_armed_bandit":
        arm      = mab_agent.select_arm()
        system_p = mab_agent.get_system_prompt(arm)
        user_p   = _env.task
        if prev:
            user_p = (
                f"Task: {_env.task}\n\n"
                f"Previous attempt ({prev['algorithm']}) scored "
                f"{prev['score']*100:.0f}% and was insufficient:\n"
                f"\"{prev['response'][:200]}...\"\n\n"
                f"Now improve using the '{arm.replace('_',' ')}' strategy."
            )
        action_type = "change_strategy"

    else:  # q_learning
        q_action = ql_agent.select_action(attempt, prev["score"] if prev else None)
        system_p = ql_agent.get_system_prompt(q_action, _env.attempts)
        user_p   = (
            f"Task: {_env.task}\n\n"
            f"Q-table selected optimal action: '{q_action.replace('_',' ')}'. "
            f"Execute this strategy for maximum quality."
        )
        action_type = "submit"

    # ── Generate response ─────────────────────────────────────
    t0       = time.time()
    response = llm(system_p, user_p)

    # ── Evaluate quality ──────────────────────────────────────
    eval_res = evaluate(response, _env.task, algo["level"])
    score    = min(1.0, max(0.0, eval_res["score"]))
    reason   = eval_res["reason"]
    passed   = score >= algo["threshold"]

    # ── Compute reward ────────────────────────────────────────
    base       = algo["reward_pass"] if passed else algo["reward_fail"]
    step_pen   = attempt * algo["step_penalty_per_attempt"]
    adapt_bon  = 0.0
    if prev and score > prev["score"]:
        adapt_bon = (score - prev["score"]) * 0.7  # adaptation bonus

    total_rew  = round(base - step_pen + adapt_bon, 4)

    # ── Update agent internals ────────────────────────────────
    if algo_key == "contextual_bandit":
        cb_agent.update(_env.task, "direct_answer", total_rew)
    elif algo_key == "multi_armed_bandit" and arm:
        mab_agent.update(arm, total_rew)
    elif algo_key == "q_learning" and q_action:
        ql_agent.update(attempt, prev["score"] if prev else None,
                        q_action, total_rew, score)

    # ── Build result objects ──────────────────────────────────
    obs = Observation(
        task=_env.task,
        attempt_number=attempt,
        algorithm=algo["name"],
        level=algo["level"],
        previous_score=prev["score"] if prev else None,
        previous_response=prev["response"][:150] if prev else None,
        context={"arm": arm, "q_action": q_action},
    )
    action = Action(
        type=action_type,
        content=response[:120] + "…",
        strategy=algo["name"].lower().replace(" ", "_"),
        arm=arm,
        q_action=q_action,
    )
    reward = Reward(
        base=round(base, 4),
        adaptation_bonus=round(adapt_bon, 4),
        step_penalty=round(step_pen, 4),
        total=total_rew,
        passed=passed,
        score=round(score, 4),
        threshold=algo["threshold"],
        reason=reason,
    )

    record = {
        "algorithm": algo["name"],
        "level": algo["level"],
        "algo_key": algo_key,
        "response": response,
        "score": score,
        "reward": total_rew,
        "passed": passed,
        "reason": reason,
        "arm": arm,
        "q_action": q_action,
        "adapt_bonus": adapt_bon,
    }
    _env.attempts.append(record)

    # ── Advance state ─────────────────────────────────────────
    if passed or _env.algo_idx == 2:
        _env.done = True
    else:
        _env.algo_idx += 1

    return StepResult(
        run_id=_env.run_id,
        step=attempt,
        observation=obs,
        action=action,
        reward=reward,
        response=response,
        algorithm=algo["name"],
        level=algo["level"],
        algo_key=algo_key,
        done=_env.done,
        timestamp=time.time(),
    )


@app.post("/run", response_model=RunResult)
def run_full(req: RunRequest):
    """
    Run the full self-improvement pipeline in one call.
    Tries CB → MAB → QL, stopping when quality threshold is met.
    """
    _env.reset(req.task)
    t0 = time.time()
    steps: List[StepResult] = []

    for _ in range(3):
        if _env.done:
            break
        result = step()
        steps.append(result)
        if result.done:
            break

    if not steps:
        raise HTTPException(500, "No steps completed.")

    total_rew   = round(sum(s.reward.total for s in steps), 4)
    last        = steps[-1]
    first_score = steps[0].reward.score
    final_score = last.reward.score
    adapt_gain  = round(max(0.0, final_score - first_score), 4)

    return RunResult(
        run_id=_env.run_id,
        task=req.task,
        steps=steps,
        total_reward=total_rew,
        winning_algorithm=last.algorithm,
        winning_algo_key=last.algo_key,
        final_score=final_score,
        adaptation_gain=adapt_gain,
        attempts_used=len(steps),
        success=last.reward.passed,
        duration_seconds=round(time.time() - t0, 2),
    )
