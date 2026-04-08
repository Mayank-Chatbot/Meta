"""
inference.py — OpenEnv baseline inference script for MetaMind
Mandatory structured stdout format: [START], [STEP], [END]
Uses OpenAI client with API_BASE_URL / MODEL_NAME / HF_TOKEN env vars.
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────
# Config from environment variables (required by submission rules)
# ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.anthropic.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
HF_TOKEN     = os.getenv("HF_TOKEN", os.getenv("ANTHROPIC_API_KEY", "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"))
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK = "MetaMind-OpenEnv"

# ─────────────────────────────────────────────────────────────────
# OpenAI-compatible client (required by submission rules)
# ─────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ─────────────────────────────────────────────────────────────────
# Tasks to evaluate (easy / medium / hard)
# ─────────────────────────────────────────────────────────────────
TASKS = [
    {
        "id": "easy",
        "task": "What is 15 multiplied by 8? Give a direct numerical answer.",
    },
    {
        "id": "medium",
        "task": "Explain the difference between supervised learning and unsupervised learning, with one example of each.",
    },
    {
        "id": "hard",
        "task": (
            "Describe the Q-Learning algorithm in detail: explain the Bellman equation, "
            "the role of learning rate (alpha), discount factor (gamma), and epsilon-greedy "
            "exploration. Include a concrete tabular update example."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def env_reset(task: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_run(task: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/run",
        json={"task": task},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def agent_answer(task: str) -> str:
    """Generate an answer using the OpenAI-compatible client."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Answer the user's question "
                    "accurately and thoroughly."
                ),
            },
            {"role": "user", "content": task},
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────
# Main — mandatory [START] / [STEP] / [END] log format
# ─────────────────────────────────────────────────────────────────

def run_inference():
    for task_def in TASKS:
        task_id = task_def["id"]
        task    = task_def["task"]

        # ── [START] ──────────────────────────────────────────
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        reward     = 0.0
        last_score = 0.0
        step_n     = 0
        done       = False

        try:
            env_reset(task)

            try:
                answer = agent_answer(task)
            except Exception as e:
                answer = f"[Agent error: {e}]"

            step_n = 1
            try:
                result     = env_run(task)
                reward     = result.get("total_reward", 0.0)
                last_score = result.get("final_score", 0.0)
                done       = result.get("success", False)
                error      = None
            except Exception as e:
                error = str(e)

            action_json = json.dumps({"answer": answer[:120] + "..."})

            # ── [STEP] ───────────────────────────────────────
            print(
                f"[STEP] step={step_n} action={action_json} "
                f"reward={reward:.2f} done={str(done).lower()} error={error}",
                flush=True,
            )

            # ── [END] ────────────────────────────────────────
            print(
                f"[END] success={str(done).lower()} steps={step_n} "
                f"score={last_score:.3f} rewards={reward:.2f}",
                flush=True,
            )

        except Exception as e:
            print(f"[END] success=false steps={step_n} score=0.000 rewards=0.00", flush=True)
            print(f"[ERROR] task={task_id} error={e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run_inference()
