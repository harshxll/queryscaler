"""
Queryscaler inference runner — LLM-driven policy, no hardcoded task plans.

The model receives the full observation (task description, schema, last result,
progress) and decides the next action autonomously using a ReAct-style prompt.
Rewards accumulate as the model iteratively improves its optimization strategy.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

try:
    from queryscaler.models import QueryscalerAction  # type: ignore
    from queryscaler.server.queryscaler_environment import QueryscalerEnvironment  # type: ignore
except ModuleNotFoundError:
    from models import QueryscalerAction  # type: ignore
    from server.queryscaler_environment import QueryscalerEnvironment  # type: ignore


# ── Environment configuration (mandatory) ────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama3-70b-8192")
API_KEY = os.getenv("API_KEY")

BENCHMARK               = "queryscaler"
TEMPERATURE             = 0.0
MAX_TOKENS              = 512
SUCCESS_SCORE_THRESHOLD = 0.5

GRADE_LEVELS: List[Tuple[int, str, int]] = [
    (0, "easy_index",     10),
    (1, "medium_rewrite", 15),
    (2, "hard_workload",  20),
]

# ── System prompt ─────────────────────────────────────────────────────────────
# FIX 1: Explicitly name the high-value actions that raise reward.
# The old prompt didn't tell the agent WHICH actions move the needle.
# Now it prioritises index creation and query rewrites over EXPLAIN loops.
SYSTEM_PROMPT = """\
You are a SQL query optimization agent working on a DuckDB database.
Your reward increases when you take CONCRETE optimization actions, not just when
you inspect. EXPLAIN is useful once — do not repeat it.

You MUST respond with a single JSON object — no markdown, no explanation, no extra text.

Available action types:
  - execute  : Run any SQL statement (DESCRIBE, CREATE INDEX, CREATE VIEW, SELECT, etc.)
  - explain  : Run EXPLAIN on a SELECT query to inspect its query plan
  - finish   : Signal that you are done optimizing

JSON schema (pick exactly one):
  {"action_type": "execute", "sql": "<sql statement>"}
  {"action_type": "explain", "sql": "<select query>"}
  {"action_type": "finish"}

REWARD RULES — read carefully:
  - Creating a composite index on WHERE/JOIN/ORDER BY columns gives HIGH reward.
  - Rewriting a query to remove SELECT *, add column lists, use CTEs, gives HIGH reward.
  - Creating a VIEW or pre-aggregation table gives HIGH reward.
  - Running EXPLAIN once to understand the plan is OK — repeating it gives ZERO extra reward.
  - Repeating any SQL you already ran gives a penalty — never repeat the same statement.
  - Call finish only after you have taken at least 3 meaningful optimization actions.

Recommended action sequence:
  1. DESCRIBE <table>  — inspect schema ONCE.
  2. EXPLAIN <baseline_query>  — inspect plan ONCE.
  3. CREATE INDEX idx_name ON table(col1, col2)  — this is the single most impactful action.
  4. Rewrite the slow query: remove SELECT *, add explicit columns, push filters early.
  5. Optionally CREATE VIEW or WITH cte AS (...) for repeated access patterns.
  6. Call finish.
"""


# ── Logging helpers ───────────────────────────────────────────────────────────
def _one_line(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = _one_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={_one_line(action)} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={max(0.0001, min(0.9999, score)):.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation → prompt ──────────────────────────────────────────────────────
def build_user_prompt(
    obs: Any,
    step: int,
    max_steps: int,
    history: List[Dict[str, str]],
) -> str:
    """
    Construct the user-turn message from the current observation so the LLM
    has full context to decide its next action.

    FIX 2: History now shows reward CHANGE per step (delta), not just the raw
    reward. This makes it obvious to the LLM which actions raised the reward
    and which did nothing — so it can learn within the episode which moves work.
    """
    lines = [
        f"=== Step {step}/{max_steps} | Current reward: {float(obs.progress):.4f} ===",
        "",
        "## Task",
        obs.task_description,
        "",
        f"## Available tables: {', '.join(obs.tables)}",
    ]

    if obs.last_result is not None:
        status = "ERROR" if (obs.last_exit_code is not None and obs.last_exit_code != 0) else "OK"
        lines += [
            "",
            f"## Last action result [{status}]",
            str(obs.last_result)[:800],
        ]

    if obs.hints:
        lines += ["", "## Hints", *obs.hints]

    if history:
        lines += ["", "## Actions taken so far (action → reward delta)"]
        prev_reward = 0.0
        # FIX 2 cont: show last 8 steps (was 6) and include reward delta
        for i, h in enumerate(history[-8:], 1):
            curr_reward = float(h["reward"])
            delta = curr_reward - prev_reward
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            lines.append(f"  {i}. {h['action']}  →  reward={curr_reward:.3f} ({delta_str})")
            prev_reward = curr_reward

    # FIX 3: Explicit nudge when agent is stuck in non-index actions.
    # If the last 3 actions were all explain or describe, push the agent forward.
    if len(history) >= 3:
        last_three = [h["action"] for h in history[-3:]]
        all_inspect = all(
            ('"action_type":"explain"' in a or '"DESCRIBE"' in a.upper() or '"describe"' in a)
            for a in last_three
        )
        if all_inspect:
            lines += [
                "",
                "## ⚠ WARNING: You have spent the last 3 steps only inspecting.",
                "   Inspection gives NO additional reward. You MUST now take a concrete",
                "   optimization action: CREATE INDEX, rewrite the SELECT, or CREATE VIEW.",
            ]

    lines += [
        "",
        "Decide your next action. Respond with a single JSON object only.",
    ]

    return "\n".join(lines)


# ── LLM action selection ──────────────────────────────────────────────────────
def choose_action(
    client: OpenAI,
    obs: Any,
    step: int,
    max_steps: int,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Ask the LLM for the next action given the current observation.
    Falls back to {"action_type": "finish"} if the response cannot be parsed.
    """
    user_prompt = build_user_prompt(obs, step, max_steps, history)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content or ""
        action_dict = json.loads(raw)

        if "action_type" not in action_dict:
            raise ValueError("Missing action_type in LLM response")

        # FIX 4: Deduplicate — if the LLM is about to repeat an action it already
        # took, force it to finish instead of burning steps on penalty actions.
        action_str = json.dumps(action_dict, separators=(",", ":"))
        past_actions = {h["action"] for h in history}
        if action_str in past_actions:
            # Repeated action — try to steer toward finish if min steps met,
            # otherwise skip this action by returning a no-op describe.
            steps_so_far = len(history)
            min_finish = [3, 4, 5]  # matches env._min_finish_steps()
            task_min = min_finish[0]  # conservative; env will enforce the real min
            if steps_so_far >= task_min:
                return {"action_type": "finish"}
            # Otherwise return a cheap unique action so the step isn't wasted.
            return {"action_type": "execute", "sql": "SELECT COUNT(*) FROM " + (obs.tables[0] if obs.tables else "orders")}

        return action_dict

    except Exception:
        return {"action_type": "finish"}


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(
    client: OpenAI,
    task_index: int,
    task_name: str,
    max_steps: int,
) -> None:
    env = QueryscalerEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.01
    success     = False
    history: List[Dict[str, str]] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env._task_index = task_index
        obs = env.reset()

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            action_dict = choose_action(client, obs, step, max_steps, history)
            action_str  = json.dumps(action_dict, separators=(",", ":"))

            # Ensure action is well-formed before passing to env
            try:
                action = QueryscalerAction(**action_dict)
            except Exception:
                action_dict = {"action_type": "finish"}
                action_str  = json.dumps(action_dict, separators=(",", ":"))
                action      = QueryscalerAction(**action_dict)

            obs = env.step(action)

            reward = float(obs.reward or 0.0)
            done   = bool(obs.done)
            error  = (
                obs.last_result
                if (obs.last_exit_code is not None and obs.last_exit_code != 0)
                else None
            )

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            rewards.append(reward)
            steps_taken = step
            final_score = float(obs.progress)
            # Store reward as float string for delta calculation in build_user_prompt
            history.append({"action": action_str, "reward": f"{reward:.4f}"})

            if done:
                break

        final_score = max(0.0001, min(0.9999, final_score))
        success     = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        finally:
            log_end(
                success=success,
                steps=steps_taken,
                score=final_score,
                rewards=rewards,
            )


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if (API_BASE_URL and API_KEY) else None
    for task_index, task_name, max_steps in GRADE_LEVELS:
        run_episode(client, task_index, task_name, max_steps)


if __name__ == "__main__":
    main()