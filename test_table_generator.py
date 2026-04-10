"""
Quick test for TableGenerator class.
Run: python test_table_generator.py
"""

import os
import tempfile
import duckdb
import sys

try:
    from queryscaler.table_generator import TableGenerator  # type: ignore
except ModuleNotFoundError:
    # If running from inside the queryscaler folder, add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from queryscaler.table_generator import TableGenerator # type: ignore


def test_easy_preset():
    print("Testing easy preset (5k rows)...")
    df = TableGenerator.easy_orders_table()
    assert len(df) == 5000, f"Expected 5000 rows, got {len(df)}"
    assert list(df.columns) == ["order_id", "customer_id", "amount", "order_date"]
    assert df["order_id"].is_unique
    print("  ✅ Easy preset passed")


def test_medium_preset():
    print("Testing medium preset (50k rows)...")
    df = TableGenerator.medium_sales_table()
    assert len(df) == 50000, f"Expected 50000 rows, got {len(df)}"
    assert list(df.columns) == ["sale_id", "customer_id", "product", "amount", "sale_date"]
    print("  ✅ Medium preset passed")


def test_hard_preset():
    print("Testing hard preset (200k rows)...")
    df = TableGenerator.hard_events_table()
    assert len(df) == 200000, f"Expected 200000 rows, got {len(df)}"
    assert list(df.columns) == ["event_id", "user_id", "event_type", "value", "timestamp", "region"]
    print("  ✅ Hard preset passed")


def test_custom_config():
    print("Testing custom config...")
    gen = TableGenerator(seed=123)
    config = {
        "rows": 100,
        "columns": {
            "id": {"type": "int", "unique": True, "start": 1},
            "score": {"type": "float", "min": 0, "max": 100, "distribution": "normal"},
            "category": {"type": "varchar", "cardinality": 5, "prefix": "cat_"},
            "active": {"type": "bool", "true_prob": 0.7},
        },
        "null_probability": 0.1,
    }
    df = gen.generate_from_config(config)
    assert len(df) == 100
    assert list(df.columns) == ["id", "score", "category", "active"]
    assert df["id"].is_unique
    # Check nulls roughly present
    null_count = df["score"].isna().sum()
    assert 5 <= null_count <= 20, f"Expected ~10 nulls, got {null_count}"
    print("  ✅ Custom config passed")


def test_duckdb_save_and_query():
    print("Testing DuckDB save and query...")
    df = TableGenerator.easy_orders_table(num_rows=1000)
    gen = TableGenerator()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        gen.save_to_duckdb(df, db_path, "orders")
        
        # New connection to the same file
        conn = duckdb.connect(db_path)
        result = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert result == 1000, f"Expected 1000 rows, got {result}"
        
        result = conn.execute(
            "SELECT AVG(amount) FROM orders WHERE customer_id = 42"
        ).fetchone()[0]
        print(f"  Average amount for customer 42: {result:.2f}")
        conn.close()
        print("  ✅ DuckDB save and query passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Running TableGenerator tests...")
    print("=" * 50)

    test_easy_preset()
    test_medium_preset()
    test_hard_preset()
    test_custom_config()
    test_duckdb_save_and_query()

    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)