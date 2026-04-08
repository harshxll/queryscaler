#!/usr/bin/env python3
"""
Local debug script: QueryscalerEnvironment + real LLM (no server).
Shows full observation, LLM response, and environment changes.
"""

import os
import sys
import json
import re
from typing import Dict, Any

from openai import OpenAI
from queryscaler.server.queryscaler_environment import QueryscalerEnvironment
from queryscaler.models import QueryscalerAction

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2")
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
   API_KEY = "ollama"

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
    main()