import os

import duckdb

try:
    from queryscaler.server.queryscaler_environment import QueryscalerEnvironment  # type: ignore
    from queryscaler.models import QueryscalerAction  # type: ignore
except ModuleNotFoundError:
    from server.queryscaler_environment import QueryscalerEnvironment  # type: ignore
    from models import QueryscalerAction  # type: ignore


class TinyQueryscalerEnvironment(QueryscalerEnvironment):
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


def test_hints_disabled_by_default():
    old_value = os.environ.pop("QUERYSCALER_ENABLE_HINTS", None)
    env = TinyQueryscalerEnvironment()
    try:
        obs = env.reset()
        assert obs.hints == []
    finally:
        env.close()
        if old_value is not None:
            os.environ["QUERYSCALER_ENABLE_HINTS"] = old_value


def test_hints_enabled_with_env_var():
    old_value = os.environ.get("QUERYSCALER_ENABLE_HINTS")
    os.environ["QUERYSCALER_ENABLE_HINTS"] = "1"
    env = TinyQueryscalerEnvironment()
    try:
        obs = env.reset()
        assert len(obs.hints) > 0
        assert any("Steps left" in hint for hint in obs.hints)
    finally:
        env.close()
        if old_value is None:
            os.environ.pop("QUERYSCALER_ENABLE_HINTS", None)
        else:
            os.environ["QUERYSCALER_ENABLE_HINTS"] = old_value


def test_hard_task_enforces_two_index_limit():
    old_value = os.environ.pop("QUERYSCALER_ENABLE_HINTS", None)
    env = TinyQueryscalerEnvironment()
    try:
        env._task_index = 2
        env.reset()

        env.step(QueryscalerAction(action_type="execute", sql="CREATE INDEX idx_1 ON events(user_id)"))
        env.step(QueryscalerAction(action_type="execute", sql="CREATE INDEX idx_2 ON events(event_type)"))
        obs = env.step(QueryscalerAction(action_type="execute", sql="CREATE INDEX idx_3 ON events(value)"))

        assert obs.last_exit_code == 1
        assert obs.last_result is not None
        assert "at most 2 indexes" in obs.last_result.lower()
    finally:
        env.close()
        if old_value is not None:
            os.environ["QUERYSCALER_ENABLE_HINTS"] = old_value


def test_reward_non_negative_after_optimization_attempt():
    old_value = os.environ.pop("QUERYSCALER_ENABLE_HINTS", None)
    env = TinyQueryscalerEnvironment()
    try:
        obs_before = env.reset()
        obs_after = env.step(
            QueryscalerAction(
                action_type="execute",
                sql="CREATE INDEX idx_orders_customer ON orders(customer_id, order_date, amount)",
            )
        )
        assert obs_after.progress >= 0.0
        assert obs_after.progress <= 1.0
        assert obs_after.last_exit_code == 0
    finally:
        env.close()
        if old_value is not None:
            os.environ["QUERYSCALER_ENABLE_HINTS"] = old_value


def run_all_tests():
    tests = [
        test_hints_disabled_by_default,
        test_hints_enabled_with_env_var,
        test_hard_task_enforces_two_index_limit,
        test_reward_non_negative_after_optimization_attempt,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL: {test.__name__} -> {exc}")
            raise
    print(f"\\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_all_tests()
