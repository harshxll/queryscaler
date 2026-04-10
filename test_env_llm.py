#!/usr/bin/env python3
"""
Local debug script: QueryscalerEnvironment + real LLM (no server).
Shows full observation, LLM response, and environment changes.
"""

import os
import sys
import json
import re
import statistics
from typing import Dict, Any, List, Tuple

from openai import OpenAI
import duckdb

try:
    from queryscaler.server.queryscaler_environment import QueryscalerEnvironment  # type: ignore
    from queryscaler.models import QueryscalerAction  # type: ignore
except ModuleNotFoundError:
    from server.queryscaler_environment import QueryscalerEnvironment  # type: ignore
    from models import QueryscalerAction  # type: ignore
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
# MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2")
# API_KEY = os.environ.get("OPENAI_API_KEY")
# if not API_KEY:
#    API_KEY = "ollama"

API_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"   # or "mixtral-8x7b-32768"
API_KEY = os.environ.get("GROQ_API_KEY")
MAX_STEPS = 8

# ------------------------------------------------------------------
# Prompt (same as inference)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a DuckDB query optimization agent.

════════════════════════════════════════
VALID ACTIONS — ONLY THESE 3 EXIST:
════════════════════════════════════════

1. Execute any SQL statement:
   {"action_type": "execute", "sql": "CREATE INDEX idx ON orders(customer_id, order_date, amount)"}

2. Get query plan (DO NOT write EXPLAIN in the sql field — it is added automatically):
   {"action_type": "explain", "sql": "SELECT * FROM orders WHERE customer_id = 42"}

3. End the episode:
   {"action_type": "finish"}

❌ INVALID — will score 0 and waste a step:
   {"action_type": "create index", ...}   ← wrong, use "execute"
   {"action_type": "explain", "sql": "EXPLAIN SELECT ..."}  ← double EXPLAIN, causes error

════════════════════════════════════════
YOUR STRATEGY (follow this exactly):
════════════════════════════════════════

Step 1 → execute: DESCRIBE orders
Step 2 → execute: CREATE INDEX idx_composite ON orders(customer_id, order_date, amount)
Step 3 → finish

The baseline query filters on customer_id, order_date, and sorts by amount.
A composite index on those 3 columns is the correct optimization.
Do not waste steps on EXPLAIN — go straight to CREATE INDEX then finish.

Reward = max(0, speedup - 1.0). A 2x speedup = score 1.0.
"""
def build_user_message(obs: Any, step: int, prev_progress: float = 0.0) -> str:
    delta = obs.progress - prev_progress
    delta_str = f"{delta:+.3f}" if step > 1 else "N/A (first step)"

    error_note = ""
    if obs.last_result:
        r = obs.last_result
        if "syntax error" in r and "EXPLAIN" in r:
            error_note = (
                "\n❌ ERROR: You wrote EXPLAIN inside the sql field of an explain action."
                "\n   Fix: {\"action_type\": \"explain\", \"sql\": \"SELECT * FROM orders WHERE ...\"}"
                "\n   The system adds EXPLAIN automatically. Never write it yourself."
            )
        elif "Unknown action_type" in r:
            error_note = (
                "\n❌ ERROR: Invalid action_type used. ONLY valid values are: execute, explain, finish."
                "\n   To create an index use: {\"action_type\": \"execute\", \"sql\": \"CREATE INDEX ...\"}"
            )
        elif "already exists" in r:
            error_note = (
                "\n❌ ERROR: Index already exists. Either call finish (optimization done)"
                "\n   or try: DROP INDEX idx_name; then recreate with different columns."
            )
        elif "Error" in r or "error" in r:
            error_note = f"\n❌ LAST ACTION FAILED: {r[:150]}"

    progress_note = ""
    if step > 1:
        if delta > 0.05:
            progress_note = "\n✅ Score improved! Call finish to lock in your score."
        elif delta <= 0 and obs.progress <= 0:
            progress_note = (
                "\n⚠️  Score is 0. Stop using explain. Execute a CREATE INDEX now:\n"
                "   {\"action_type\": \"execute\", \"sql\": \"CREATE INDEX idx ON orders(customer_id, order_date, amount)\"}"
            )

    hints_str = "\n".join(f"  • {h}" for h in obs.hints)

    return (
        f"Step {step}/{MAX_STEPS} | Score: {obs.progress:.3f} (Δ {delta_str})\n"
        f"Tables: {obs.tables}\n"
        f"Last result: {obs.last_result or 'N/A'}"
        f"{error_note}"
        f"{progress_note}\n"
        f"Hints:\n{hints_str}\n\n"
        f"Respond with ONE JSON action (action_type must be execute, explain, or finish):"
    )

def extract_json(raw: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from LLM output."""
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()
    text = text.replace("```", "").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                return json.loads(candidate)
    raise ValueError("Unbalanced braces")


class TinyQueryscalerEnvironment(QueryscalerEnvironment):
    """Small synthetic dataset for quick reward scenario tests."""

    def _measure_baseline_time(self, query: str, runs: int = 1) -> float:
        return self._measure_query_time(query)

    def _stable_query_time(self, query: str, runs: int = 1) -> float:
        return self._measure_query_time(query)

    def _setup_task_data(self):
        conn = duckdb.connect(self.db_path)
        try:
            if self._task_index == 0:
                conn.execute(
                    """
                    CREATE TABLE orders AS
                    SELECT
                        i AS order_id,
                        CASE WHEN i % 5 = 0 THEN 42 ELSE i % 100 END AS customer_id,
                        CAST(100 + (i % 1000) AS DOUBLE) AS amount,
                        DATE '2024-01-01' + CAST(i % 120 AS INTEGER) AS order_date
                    FROM range(1, 5001) t(i)
                    """
                )
                self._baseline_query = (
                    "SELECT * FROM orders "
                    "WHERE customer_id = 42 AND order_date > '2024-01-01' "
                    "ORDER BY amount DESC LIMIT 100"
                )
                self.task_description = "Tiny easy task"
            elif self._task_index == 1:
                conn.execute(
                    """
                    CREATE TABLE sales AS
                    SELECT
                        i AS sale_id,
                        i % 100 AS customer_id,
                        i % 500 AS product,
                        CAST((i % 1000) AS DOUBLE) AS amount,
                        DATE '2024-06-01' + CAST(i % 60 AS INTEGER) AS sale_date
                    FROM range(1, 5001) t(i)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE products AS
                    SELECT i AS id, i % 30 AS category, CAST(10 + (i % 200) AS DOUBLE) AS price
                    FROM range(0, 500) t(i)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE categories AS
                    SELECT i AS category, 'Category_' || i::VARCHAR AS name
                    FROM range(0, 30) t(i)
                    """
                )
                self._baseline_query = (
                    "SELECT s.sale_id, s.amount, p.price, c.name "
                    "FROM sales s "
                    "JOIN products p ON s.product = p.id "
                    "JOIN categories c ON p.category = c.category "
                    "WHERE s.sale_date >= '2024-06-01' AND s.amount > 500"
                )
                self.task_description = "Tiny medium task"
            else:
                conn.execute(
                    """
                    CREATE TABLE events AS
                    SELECT
                        i AS event_id,
                        (i % 20) + 1 AS user_id,
                        'type_' || (i % 10)::VARCHAR AS event_type,
                        CAST(i % 500 AS DOUBLE) AS value,
                        DATE '2024-06-01' + CAST(i % 30 AS INTEGER) AS timestamp,
                        'r' || (i % 4)::VARCHAR AS region
                    FROM range(1, 8001) t(i)
                    """
                )
                self._workload_queries = [
                    "SELECT user_id, COUNT(*) FROM events WHERE event_type = 'type_5' GROUP BY user_id",
                    "SELECT * FROM events WHERE user_id = 7 AND value > 100",
                    "SELECT event_type, AVG(value) FROM events WHERE timestamp >= '2024-06-01' GROUP BY event_type",
                ]
                self._baseline_query = self._workload_queries[0]
                self.task_description = "Tiny hard task"
        finally:
            conn.close()

        if self._task_index < 2:
            self._evaluation_queries = [self._baseline_query]
        else:
            self._evaluation_queries = list(self._workload_queries)

        self._baseline_times = {
            query: self._measure_baseline_time(query, runs=2)
            for query in self._evaluation_queries
        }
        self._baseline_time = sum(self._baseline_times.values()) / len(self._baseline_times)


def run_action_sequence(env: QueryscalerEnvironment, actions: List[Dict[str, Any]]) -> List[float]:
    """Run a fixed action sequence and return reward history."""
    rewards: List[float] = []
    env.reset()
    for action_dict in actions:
        obs = env.step(QueryscalerAction(**action_dict))
        rewards.append(obs.reward)
        if obs.done:
            break
    return rewards


def run_reward_validation_suite(episodes: int = 3) -> None:
    """
    Reward validation scenarios (no external LLM needed).
    Focuses on stability/convergence and behavior under different strategies.
    """
    print("=" * 70)
    print("REWARD VALIDATION SUITE")
    print("=" * 70)

    scenario_results: List[Tuple[str, bool, str]] = []

    # Scenario 1: No optimization should keep rewards relatively low.
    env = TinyQueryscalerEnvironment()
    try:
        no_opt_actions = [
            {"action_type": "execute", "sql": "DESCRIBE orders"},
            {"action_type": "explain", "sql": "SELECT * FROM orders WHERE customer_id = 42"},
            {"action_type": "finish"},
        ]
        rewards = run_action_sequence(env, no_opt_actions)
        final_reward = rewards[-1] if rewards else 0.0
        passed = final_reward <= 0.55
        scenario_results.append(
            ("No optimization stays low", passed, f"final_reward={final_reward:.3f}")
        )
    finally:
        env.close()

    # Scenario 2: Index optimization should beat no-op strategy.
    env = TinyQueryscalerEnvironment()
    try:
        index_actions = [
            {"action_type": "execute", "sql": "DESCRIBE orders"},
            {
                "action_type": "execute",
                "sql": "CREATE INDEX idx_composite ON orders(customer_id, order_date, amount)",
            },
            {"action_type": "finish"},
        ]
        rewards = run_action_sequence(env, index_actions)
        final_reward = rewards[-1] if rewards else 0.0
        passed = final_reward >= 0.35
        scenario_results.append(
            ("Index optimization improves reward", passed, f"final_reward={final_reward:.3f}")
        )
    finally:
        env.close()

    # Scenario 3: Repetition should trigger penalty and avoid reward inflation.
    env = TinyQueryscalerEnvironment()
    try:
        repetitive_actions = [
            {"action_type": "execute", "sql": "DESCRIBE orders"},
            {"action_type": "execute", "sql": "DESCRIBE orders"},
            {"action_type": "execute", "sql": "DESCRIBE orders"},
            {"action_type": "finish"},
        ]
        rewards = run_action_sequence(env, repetitive_actions)
        final_reward = rewards[-1] if rewards else 0.0
        passed = final_reward <= 0.45
        scenario_results.append(
            ("Repetition does not inflate reward", passed, f"final_reward={final_reward:.3f}")
        )
    finally:
        env.close()

    # Scenario 4: Hard-task index limit should be enforced.
    env = TinyQueryscalerEnvironment()
    try:
        env._task_index = 2
        env.reset()
        env.step(QueryscalerAction(action_type="execute", sql="CREATE INDEX idx_1 ON events(user_id)"))
        env.step(QueryscalerAction(action_type="execute", sql="CREATE INDEX idx_2 ON events(event_type)"))
        obs = env.step(QueryscalerAction(action_type="execute", sql="CREATE INDEX idx_3 ON events(value)"))
        passed = obs.last_exit_code == 1 and (obs.last_result is not None and "at most 2 indexes" in obs.last_result.lower())
        detail = obs.last_result[:100] if obs.last_result else "no result"
        scenario_results.append(("Hard task enforces 2-index cap", passed, detail))
    finally:
        env.close()

    # Scenario 5: Convergence/stability across repeated optimized episodes.
    optimized_rewards: List[float] = []
    for _ in range(episodes):
        env = TinyQueryscalerEnvironment()
        try:
            rewards = run_action_sequence(
                env,
                [
                    {"action_type": "execute", "sql": "DESCRIBE orders"},
                    {
                        "action_type": "execute",
                        "sql": "CREATE INDEX idx_composite ON orders(customer_id, order_date, amount)",
                    },
                    {"action_type": "finish"},
                ],
            )
            optimized_rewards.append(rewards[-1] if rewards else 0.0)
        finally:
            env.close()

    mean_reward = statistics.mean(optimized_rewards) if optimized_rewards else 0.0
    stdev_reward = statistics.pstdev(optimized_rewards) if len(optimized_rewards) > 1 else 0.0
    # A stable scorer should not fluctuate heavily for the same policy.
    converged = stdev_reward <= 0.15
    scenario_results.append(
        (
            "Optimized policy reward is stable",
            converged,
            f"mean={mean_reward:.3f}, stdev={stdev_reward:.3f}, samples={optimized_rewards}",
        )
    )

    # Scenario 6: Different long-horizon medium-task optimization (>3 steps).
    env = TinyQueryscalerEnvironment()
    try:
        env._task_index = 1
        long_actions = [
            {"action_type": "execute", "sql": "DESCRIBE sales"},
            {
                "action_type": "explain",
                "sql": (
                    "SELECT s.sale_id, s.amount, p.price, c.name "
                    "FROM sales s "
                    "JOIN products p ON s.product = p.id "
                    "JOIN categories c ON p.category = c.category "
                    "WHERE s.sale_date >= '2024-06-01' AND s.amount > 500"
                ),
            },
            {
                "action_type": "execute",
                "sql": "CREATE INDEX idx_sales_date_amt_prod ON sales(sale_date, amount, product)",
            },
            {
                "action_type": "execute",
                "sql": "CREATE INDEX idx_products_id_cat ON products(id, category)",
            },
            {
                "action_type": "execute",
                "sql": (
                    "WITH filtered_sales AS ("
                    "SELECT sale_id, amount, product FROM sales "
                    "WHERE sale_date >= '2024-06-01' AND amount > 500"
                    ") "
                    "SELECT fs.sale_id, fs.amount, p.price, c.name "
                    "FROM filtered_sales fs "
                    "JOIN products p ON fs.product = p.id "
                    "JOIN categories c ON p.category = c.category"
                ),
            },
            {"action_type": "finish"},
        ]
        rewards = run_action_sequence(env, long_actions)
        first_reward = rewards[0] if rewards else 0.0
        final_reward = rewards[-1] if rewards else 0.0
        improved_overall = final_reward > first_reward + 0.10
        has_length = len(rewards) >= 5
        peak_step_gain = 0.0
        for i in range(1, len(rewards)):
            peak_step_gain = max(peak_step_gain, rewards[i] - rewards[i - 1])
        passed = has_length and improved_overall and peak_step_gain > 0.05 and final_reward >= 0.30
        scenario_results.append(
            (
                "Long-horizon medium strategy improves progressively",
                passed,
                (
                    f"steps={len(rewards)}, first={first_reward:.3f}, "
                    f"final={final_reward:.3f}, peak_gain={peak_step_gain:.3f}, rewards={rewards}"
                ),
            )
        )
    finally:
        env.close()

    # Print and enforce pass/fail summary.
    failures = 0
    for name, passed, detail in scenario_results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name} -> {detail}")
        if not passed:
            failures += 1

    print("-" * 70)
    print(f"Scenarios passed: {len(scenario_results) - failures}/{len(scenario_results)}")
    if failures > 0:
        raise SystemExit(1)
    print("Reward validation suite completed successfully.")

# ------------------------------------------------------------------
# Main debug loop
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DEBUG: QueryscalerEnvironment with Real LLM")
    print("=" * 70)

    env = QueryscalerEnvironment()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    obs = env.reset()
    print("\n--- INITIAL OBSERVATION ---")
    print(f"Task: {obs.task_description}")
    print(f"Tables: {obs.tables}")
    print(f"Progress: {obs.progress:.3f}")
    print(f"Hints: {obs.hints}")
    print("-" * 70)

    # ✅ FIX 1: Initialize history ONCE before the loop
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    step = 0
    done = False
    prev_progress = 0.0

    while not done and step < MAX_STEPS:
        step += 1
        print(f"\n>>> STEP {step} <<<")

        # ✅ FIX 2: Richer user message — shows reward delta and repeats warning
        user_msg = build_user_message(obs, step, prev_progress)

        # ✅ FIX 3: Append user message to running history (not replace it)
        messages.append({"role": "user", "content": user_msg})

        print("Calling LLM...")
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw_response = resp.choices[0].message.content or ""
            print(f"LLM Raw Response:\n{raw_response}")
        except Exception as e:
            print(f"LLM ERROR: {e}")
            break

        # ✅ FIX 4: Append assistant reply to history so next step sees it
        messages.append({"role": "assistant", "content": raw_response})

        try:
            action_dict = extract_json(raw_response)
            print(f"Parsed Action: {json.dumps(action_dict, indent=2)}")
        except Exception as e:
            print(f"JSON PARSE ERROR: {e}")
            action_dict = {"action_type": "finish"}

        try:
            action = QueryscalerAction(**action_dict)
        except Exception as e:
            print(f"Action validation error: {e}")
            action = QueryscalerAction(action_type="finish")

        prev_progress = obs.progress
        obs = env.step(action)
        done = obs.done

        print("\n--- NEW OBSERVATION ---")
        print(f"Reward: {obs.reward:.3f}  (Δ {obs.reward - prev_progress:+.3f})")
        print(f"Done: {obs.done}")
        if obs.last_result:
            preview = obs.last_result[:300] + "..." if len(obs.last_result) > 300 else obs.last_result
            print(f"Last result: {preview}")
        print("-" * 70)

        if done:
            print("\n✅ Episode finished.")
            break

    env.close()
    print(f"\nFinal score: {obs.progress:.3f}")



if __name__ == "__main__":
    if "--reward-tests" in sys.argv:
        run_reward_validation_suite()
    else:
        main()