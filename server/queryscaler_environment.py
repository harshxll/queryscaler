# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Queryscaler Environment - Query optimization sandbox with recursive cost model.
"""

import os
import time
import tempfile
import shutil
import re
import statistics
import sqlglot #type: ignore
from sqlglot import exp #type: ignore
from uuid import uuid4
from typing import List, Dict, Optional, Any, Tuple

import duckdb #type: ignore 

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Import from parent package
try:
    from ..models import QueryscalerAction, QueryscalerObservation
    from ..table_generator import TableGenerator
except ImportError:
    from models import QueryscalerAction, QueryscalerObservation #type: ignore
    from table_generator import TableGenerator #type: ignore



# ============================================================================
#  Queryscaler Environment
# ============================================================================
class QueryscalerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._enable_hints = os.environ.get("QUERYSCALER_ENABLE_HINTS", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self._task_index = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_episode()

    def _init_episode(self):
        self._optimized_query = None

        # tracking
        self._query_history = []
        self._action_counts = {}

        # strategy flags
        self._used_index = False
        self._used_cte = False
        self._used_view = False
        self._used_subquery = False
        self._schema_change = False

        # index scoring
        self._best_index_score = 0.0
        self._step = 0
        self._done = False
        self._optimization_attempted = False
        self._used_explain = False
        self._meaningful_actions = 0
        self._reward_ema = 0.0
        self._best_reward = 0.0
        self._created_indexes = set()
        self._temp_dir = tempfile.mkdtemp(prefix="queryscaler_")
        self.db_path = os.path.join(self._temp_dir, "data.db")
        self._setup_task_data()

    def _setup_task_data(self):
        gen = TableGenerator(seed=self._task_index)

        if self._task_index == 0:   # Easy – Business goal only
            df = TableGenerator.easy_orders_table(num_rows=800000)
            gen.save_to_duckdb(df, self.db_path, "orders")
            conn = duckdb.connect(self.db_path)
            conn.execute("UPDATE orders SET customer_id = 42 WHERE rowid % 5 < 2")
            conn.close()
            
            self._baseline_query = (
                "SELECT * FROM orders "
                "WHERE customer_id = 42 AND order_date > '2024-01-01' "
                "ORDER BY amount DESC LIMIT 100"
            )
            self.task_description = (
                "The customer service team frequently looks up recent high‑value orders "
                "for a specific customer (e.g., customer ID 42). The current process is slow.\n\n"
                "Your goal: Improve the performance of these lookups.\n\n"
                "You have full access to the 'orders' table. Use EXPLAIN to understand "
                "current query plans, and execute SQL to optimize. You can try composite indexing, "
                "query rewrites, reduced column scans, or precomputation strategies. "
                "You will receive reward based on measured speed and optimization quality.\n\n"
                "Start by examining the schema (execute 'DESCRIBE orders') and exploring!"
            )

        elif self._task_index == 1: # Medium – Business problem description
            df_sales = TableGenerator.medium_sales_table(num_rows=800000)
            gen.save_to_duckdb(df_sales, self.db_path, "sales")
            conn = duckdb.connect(self.db_path)
            conn.execute(
                "CREATE TABLE products AS "
                "SELECT DISTINCT product AS id, "
                "CAST(random()*500 AS INT) AS category, "
                "CAST(random()*1000 AS DECIMAL(10,2)) AS price "
                "FROM sales LIMIT 50000"
            )
            conn.execute(
                "CREATE TABLE categories AS "
                "SELECT category, 'Category_' || category AS name "
                "FROM (SELECT DISTINCT category FROM products)"
            )
            conn.close()
            
            self._baseline_query = (
                "SELECT s.sale_id, s.amount, p.price, c.name "
                "FROM sales s "
                "JOIN products p ON s.product = p.id "
                "JOIN categories c ON p.category = c.category "
                "WHERE s.sale_date >= '2024-06-01' AND s.amount > 500"
            )
            self.task_description = (
                "The sales dashboard runs a report that joins sales, products, and categories "
                "to show recent large transactions. The dashboard is timing out during peak hours.\n\n"
                "Your goal: Make this reporting query significantly faster.\n\n"
                "You have tables: 'sales', 'products', 'categories'. "
                "Use any SQL technique (indexes, query rewrites, CTEs, views, predicate pushdown, "
                "projection pruning). Compare alternatives and keep the most effective one."
            )

        else:  # Hard – Workload tuning without explicit queries
            df = TableGenerator.hard_events_table(num_rows=2000000)
            gen.save_to_duckdb(df, self.db_path, "events")
            conn = duckdb.connect(self.db_path)
            conn.execute("UPDATE events SET user_id = (user_id % 10) + 1 WHERE rowid % 3 = 0")
            conn.close()
            
            self._workload_queries = [
                "SELECT user_id, COUNT(*) FROM events WHERE event_type = 'type_5' GROUP BY user_id",
                "SELECT * FROM events WHERE user_id = 7 AND value > 100",
                "SELECT event_type, AVG(value) FROM events WHERE timestamp >= '2024-06-01' GROUP BY event_type"
            ]
            self._baseline_query = self._workload_queries[0]
            self.task_description = (
                "Your analytics team runs three important queries against the 'events' table:\n"
                "1. A report counting events per user for a specific event type.\n"
                "2. A detailed view of a single user's high‑value events.\n"
                "3. An average value aggregation for recent events.\n\n"
                "You may create AT MOST 2 INDEXES total. Your goal is to maximize the "
                "overall performance improvement across all three queries. Consider index tradeoffs, "
                "query rewrites, and balanced improvements rather than optimizing only one query.\n\n"
                "First, explore the 'events' schema and then decide which indexes to create. "
                "You can also rewrite queries if helpful."
            )

        # Baseline timings for stable, fair speed comparisons.
        if self._task_index < 2:
            self._evaluation_queries = [self._baseline_query]
        else:
            self._evaluation_queries = list(self._workload_queries)

        self._baseline_times = {
            query: self._measure_baseline_time(query)
            for query in self._evaluation_queries
        }
        self._baseline_time = statistics.mean(self._baseline_times.values())

    def _measure_query_time(self, query: str) -> float:
        conn = duckdb.connect(self.db_path)
        start = time.time()
        try:
            conn.execute(query).fetchall()
            return time.time() - start
        finally:
            conn.close()

    def _estimate_query_cost(self, query: str) -> float:
        conn = duckdb.connect(self.db_path)
        try:
            plan_df = conn.execute(f"EXPLAIN ANALYZE {query}").fetchdf()
            cost_columns = ['estimated_cost', 'total', 'cost', 'estimated_total_cost']
            for col in cost_columns:
                if col in plan_df.columns:
                    total_cost = plan_df[col].iloc[-1] if len(plan_df) > 0 else 0.0
                    return max(float(total_cost), 0.001)  # ensure positive
            # Fallback: use actual execution time (in milliseconds), ensure positive
            return max(self._measure_query_time(query) * 1000, 0.001)
        except Exception:
            return 100000.0  # high default
        finally:
            conn.close()

    def _measure_baseline_time(self, query: str, runs: int = 5) -> float:
        times = []
        for _ in range(runs):
            conn = duckdb.connect(self.db_path)
            start = time.time()
            conn.execute(query).fetchall()
            times.append(time.time() - start)
            conn.close()
        return statistics.median(times)

    def _stable_query_time(self, query: str, runs: int = 3) -> float:
        times = []
        for _ in range(runs):
            conn = duckdb.connect(self.db_path)
            start = time.time()
            conn.execute(query).fetchall()
            times.append(time.time() - start)
            conn.close()
        return statistics.median(times)

    def _extract_index_columns(self, sql):
        match = re.search(r"\((.*?)\)", sql)
        if not match:
            return []
        return [c.strip().lower() for c in match.group(1).split(",")]

    def _score_index(self, sql):
        try:
            tree = sqlglot.parse_one(self._baseline_query)
        except:
            return 0.0

        filter_cols = set()
        for col in tree.find_all(exp.Column):
            filter_cols.add(col.name.lower())

        index_cols = self._extract_index_columns(sql)

        score = 0.0
        for col in index_cols:
            if col in filter_cols:
                score += 0.3

        return min(score, 1.0)


    def _structural_score(self, query):
        try:
            tree = sqlglot.parse_one(query)
        except:
            return -0.5

        score = 0.0

        # SELECT *
        if any(isinstance(e, exp.Star) for e in tree.expressions):
            score -= 0.3
        else:
            score += 0.2

        # WHERE
        if tree.args.get("where"):
            score += 0.2

        # LIMIT
        if tree.args.get("limit"):
            score += 0.2

        # JOINS
        joins = len(list(tree.find_all(exp.Join)))
        score -= joins * 0.1

        return score


    def _strategy_score(self):
        score = 0.0

        if self._used_index:
            score += 0.3
        if self._used_cte:
            score += 0.2
        if self._used_view:
            score += 0.2
        if self._used_subquery:
            score += 0.2
        if self._schema_change:
            score += 0.3
        if self._used_explain:
            score += 0.1

        return min(score, 1.0)

    def _grade(self) -> float:
        # choose query (supports rewrites for easy/medium)
        query = self._optimized_query or self._baseline_query

        # 1) Speed component (primary signal)
        if self._task_index < 2:
            baseline_time = self._baseline_times[self._baseline_query]
            current_time = self._stable_query_time(query)
            speedup = baseline_time / max(current_time, 1e-6)
            speed_score = min(max((speedup - 1.0) / 0.5, 0.0), 1.0)

        else:
            speedups = []
            for q in self._workload_queries:
                ct = self._stable_query_time(q)
                bt = self._baseline_times[q]
                speedups.append(bt / max(ct, 1e-6))
            avg_speedup = sum(speedups) / len(speedups)
            speed_score = min(max((avg_speedup - 1.0) / 0.35, 0.0), 1.0)

        # 2) Structural score normalized to [0, 1]
        struct_raw = self._structural_score(query)
        struct_score = min(max((struct_raw + 0.5) / 1.4, 0.0), 1.0)

        # 3) Strategy score
        strat_score = self._strategy_score()

        # 4) Index quality
        index_score = self._best_index_score

        # 5) Repetition penalty
        repeat_penalty = sum(
            0.03 * (c - 1) for c in self._action_counts.values() if c > 1
        )

        # 6) Progressive reward shaping for smoother convergence.
        optimization_signal = min(
            1.0,
            0.7 * index_score +
            0.2 * strat_score +
            0.1 * struct_score,
        )
        raw_reward = 0.35 * speed_score + 0.65 * optimization_signal

        # Start low and grow as the agent performs meaningful optimization actions.
        progression = min(1.0, self._meaningful_actions / 2.0)
        target_reward = max(0.0, min(1.0, (raw_reward * progression) - repeat_penalty))

        # EMA + best-so-far tracking lowers variance and encourages convergence.
        self._reward_ema = 0.45 * target_reward + 0.55 * self._reward_ema
        self._best_reward = max(self._best_reward, self._reward_ema)

        return max(0.0, min(1.0, self._best_reward))

    def _max_steps(self) -> int:
        return [10, 15, 20][self._task_index]

    def _min_finish_steps(self) -> int:
        return [3, 4, 5][self._task_index]

    def reset(self) -> QueryscalerObservation:
        if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._init_episode()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return self._make_observation(reward=0.0, done=False)

    def step(self, action: QueryscalerAction) -> QueryscalerObservation:
        self._step += 1
        self._state.step_count += 1
        sql = action.sql
        self._last_result = None
        self._last_exit_code = 0

        try:
            if action.action_type == "execute":
                sql_lower = sql.strip().lower()

                # track repetition
                self._query_history.append(sql_lower)
                self._action_counts[sql_lower] = self._action_counts.get(sql_lower, 0) + 1

                # track optimized query
                if sql_lower.startswith("select"):
                    self._optimized_query = sql
                    if sql_lower != self._baseline_query.strip().lower():
                        self._optimization_attempted = True

                # strategy detection
                if "create index" in sql_lower:
                    self._used_index = True
                    self._meaningful_actions += 1

                if "with " in sql_lower:
                    self._used_cte = True
                    self._meaningful_actions += 1

                if "create view" in sql_lower:
                    self._used_view = True
                    self._meaningful_actions += 1

                if "select" in sql_lower and "(" in sql_lower:
                    self._used_subquery = True
                    self._meaningful_actions += 1

                if "create table" in sql_lower or "alter table" in sql_lower:
                    self._schema_change = True
                    self._meaningful_actions += 1

                if "create index" in sql_lower:
                    if self._task_index == 2:
                        match = re.search(
                            r"create\s+index\s+(?:if\s+not\s+exists\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
                            sql_lower,
                        )
                        idx_name = match.group(1) if match else None
                        if idx_name and idx_name not in self._created_indexes and len(self._created_indexes) >= 2:
                            self._last_result = "Hard task limit reached: at most 2 indexes are allowed."
                            self._last_exit_code = 1
                            reward = self._grade()
                            self._done = self._step >= self._max_steps()
                            return self._make_observation(reward=reward, done=self._done)
                        if idx_name:
                            self._created_indexes.add(idx_name)
                
                if "create index" in sql_lower:
                    score = self._score_index(sql_lower)
                    self._best_index_score = max(self._best_index_score, score)
                conn = duckdb.connect(self.db_path)
                try:
                    result = conn.execute(sql)
                    if result is not None:
                        rows = result.fetchall()
                        output = "\n".join(str(row) for row in rows[:50])
                    else:
                        output = "Statement executed successfully."
                    self._last_result = output[:1000]
                    self._last_exit_code = 0
                    if sql.upper().strip().startswith(("CREATE INDEX", "CREATE VIEW", "WITH ")):
                        self._optimization_attempted = True
                except Exception as e:
                    self._last_result = str(e)
                    self._last_exit_code = 1
                finally:
                    conn.close()
            elif action.action_type == "explain":
                conn = duckdb.connect(self.db_path)
                try:
                    plan = conn.execute(f"EXPLAIN {sql}").fetchall()
                    self._last_result = "\n".join(str(row) for row in plan)[:1000]
                    self._last_exit_code = 0
                    self._used_explain = True
                    self._meaningful_actions += 1
                except Exception as e:
                    self._last_result = f"EXPLAIN error: {e}"[:1000]
                    self._last_exit_code = 1
                finally:
                    conn.close()
            elif action.action_type == "finish":
                if self._step < self._min_finish_steps():
                    self._last_result = (
                        f"Finish is locked until step {self._min_finish_steps()} for this task. "
                        "Try more optimization actions first."
                    )
                    self._last_exit_code = 1
                    reward = max(0.0, self._grade() - 0.05)
                    self._done = False
                    return self._make_observation(reward=reward, done=False)
                reward = self._grade()
                if self._optimization_attempted:
                    reward = min(1.0, reward + 0.2)
                self._done = True
                return self._make_observation(reward=reward, done=True)
            else:
                self._last_result = f"Unknown action_type: {action.action_type}"
                self._last_exit_code = 1
        except Exception as e:
            self._last_result = str(e)
            self._last_exit_code = 1

        reward = self._grade()
        # Avoid premature termination. The agent should iterate and improve over time.
        self._done = self._step >= self._max_steps()
        return self._make_observation(reward=reward, done=self._done)

    def _make_observation(self, reward: float, done: bool) -> QueryscalerObservation:
        task_ids = ["easy_index", "medium_rewrite", "hard_workload"]
        conn = duckdb.connect(self.db_path)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        conn.close()
        
        hints = []
        if self._enable_hints:
            hints = [
                f"Steps left: {self._max_steps() - self._step}",
                "Use EXPLAIN and measured runtime to validate improvements.",
            ]
            if self._task_index == 2:
                hints.append(f"Indexes created: {len(self._created_indexes)}/2")
            if not self._optimization_attempted and self._step >= 3:
                hints.append("No optimization detected yet. Try index creation or a query rewrite.")
        
        return QueryscalerObservation(
            step_number=self._step,
            task_id=task_ids[self._task_index],
            task_description=self.task_description,
            tables=tables,
            last_result=getattr(self, '_last_result', None),
            last_exit_code=getattr(self, '_last_exit_code', None),
            progress=reward,
            hints=hints,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)