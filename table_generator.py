"""
Flexible pseudo-table generator with custom schema, size, and constraints.
Includes easy/medium/hard presets for quick task creation.
"""

import random
import numpy as np
import pandas as pd
from faker import Faker
import duckdb
from typing import Dict, List, Optional, Any


class TableGenerator:
    """
    Generate synthetic tables with full control over columns, distributions, and constraints.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)

    # ------------------------------------------------------------------
    # Presets (sized for <20 minute total inference runs)
    # ------------------------------------------------------------------
    @classmethod
    def easy_orders_table(cls, num_rows: int = 5000) -> pd.DataFrame:
        """Simple orders table: 5k rows, uniform distribution, no nulls."""
        gen = cls(seed=42)
        config = {
            "rows": num_rows,
            "columns": {
                "order_id": {"type": "int", "unique": True, "start": 1},
                "customer_id": {"type": "int", "min": 1, "max": 100, "distribution": "uniform"},
                "amount": {"type": "float", "min": 10.0, "max": 500.0, "distribution": "uniform"},
                "order_date": {"type": "date", "start": "2023-01-01", "end": "2023-12-31"},
            },
            "null_probability": 0.0,
        }
        return gen.generate_from_config(config)

    @classmethod
    def medium_sales_table(cls, num_rows: int = 50000) -> pd.DataFrame:
        """Sales table: 50k rows, skewed distribution, 3% nulls."""
        gen = cls(seed=42)
        config = {
            "rows": num_rows,
            "columns": {
                "sale_id": {"type": "int", "unique": True, "start": 1},
                "customer_id": {"type": "int", "min": 1, "max": 500, "distribution": "zipf", "alpha": 2.0},
                "product": {"type": "varchar", "cardinality": 8, "values": ["A","B","C","D","E","F","G","H"]},
                "amount": {"type": "float", "min": 1.0, "max": 1000.0, "distribution": "lognormal", "mean": 3.0, "sigma": 1.0},
                "sale_date": {"type": "date", "start": "2022-01-01", "end": "2024-12-31"},
            },
            "null_probability": 0.03,
        }
        return gen.generate_from_config(config)

    @classmethod
    def hard_events_table(cls, num_rows: int = 200000) -> pd.DataFrame:
        """Events table: 200k rows, heavy skew, 8% nulls."""
        gen = cls(seed=42)
        config = {
            "rows": num_rows,
            "columns": {
                "event_id": {"type": "int", "unique": True, "start": 1},
                "user_id": {"type": "int", "min": 1, "max": 10000, "distribution": "zipf", "alpha": 3.0},
                "event_type": {"type": "varchar", "cardinality": 30, "prefix": "type_"},
                "value": {"type": "float", "min": 0.0, "max": 5000.0, "distribution": "lognormal", "mean": 2.0, "sigma": 2.0},
                "timestamp": {"type": "timestamp", "start": "2024-01-01 00:00:00", "end": "2024-12-31 23:59:59"},
                "region": {"type": "varchar", "cardinality": 10, "values": ["North","South","East","West","Central","Northeast","Northwest","Southeast","Southwest","Unknown"]},
            },
            "null_probability": 0.08,
        }
        return gen.generate_from_config(config)

    # ------------------------------------------------------------------
    # Core generator from config dict
    # ------------------------------------------------------------------
    def generate_from_config(self, config: Dict[str, Any]) -> pd.DataFrame:
        num_rows = config["rows"]
        columns_spec = config["columns"]
        null_prob = config.get("null_probability", 0.0)

        data = {}
        for col_name, spec in columns_spec.items():
            col_type = spec.get("type", "varchar")
            values = self._generate_column(num_rows, col_type, spec)

            # Skip nulls for unique columns
            if null_prob > 0 and not spec.get("unique", False):
                null_mask = np.random.random(num_rows) < null_prob
                values = [None if m else v for m, v in zip(null_mask, values)]

            data[col_name] = values

        return pd.DataFrame(data)

    def _generate_column(self, size: int, col_type: str, spec: Dict) -> List[Any]:
        if col_type == "int":
            return self._generate_int_column(size, spec)
        elif col_type == "float":
            return self._generate_float_column(size, spec)
        elif col_type == "varchar":
            return self._generate_varchar_column(size, spec)
        elif col_type == "date":
            return self._generate_date_column(size, spec)
        elif col_type == "timestamp":
            return self._generate_timestamp_column(size, spec)
        elif col_type == "bool":
            true_prob = spec.get("true_prob", 0.5)
            return [random.random() < true_prob for _ in range(size)]
        elif col_type == "uuid":
            import uuid
            return [str(uuid.uuid4()) for _ in range(size)]
        else:
            raise ValueError(f"Unsupported column type: {col_type}")

    def _generate_int_column(self, size: int, spec: Dict) -> List[int]:
        min_val = spec.get("min", 0)
        max_val = spec.get("max")
        unique = spec.get("unique", False)
        dist = spec.get("distribution", "uniform")

        # If unique, always generate a sequential range (ignore max if too small)
        if unique:
            # If max is provided but range is insufficient, warn and extend
            if max_val is not None and (max_val - min_val + 1) < size:
                # Generate sequential starting from min_val, ignoring max
                return list(range(min_val, min_val + size))
            elif max_val is not None:
                return list(range(min_val, min_val + size))
            else:
                # No max provided, generate sequential from min_val
                return list(range(min_val, min_val + size))

        # Non-unique columns: use max_val or default
        if max_val is None:
            max_val = 100  # default

        if dist == "uniform":
            return np.random.randint(min_val, max_val + 1, size).tolist()
        elif dist == "normal":
            mean = spec.get("mean", (min_val + max_val) / 2)
            std = spec.get("std", (max_val - min_val) / 4)
            vals = np.random.normal(mean, std, size)
            vals = np.clip(vals, min_val, max_val).astype(int)
            return vals.tolist()
        elif dist == "zipf":
            alpha = spec.get("alpha", 2.0)
            vals = np.random.zipf(alpha, size)
            vals = (vals - vals.min()) / (vals.max() - vals.min())
            vals = (vals * (max_val - min_val) + min_val).astype(int)
            return vals.tolist()
        else:
            return np.random.randint(min_val, max_val + 1, size).tolist()

    def _generate_float_column(self, size: int, spec: Dict) -> List[float]:
        min_val = spec.get("min", 0.0)
        max_val = spec.get("max", 100.0)
        dist = spec.get("distribution", "uniform")

        if dist == "uniform":
            return np.round(np.random.uniform(min_val, max_val, size), 2).tolist()
        elif dist == "lognormal":
            mean = spec.get("mean", 1.0)
            sigma = spec.get("sigma", 0.5)
            vals = np.random.lognormal(mean, sigma, size)
            vals = np.clip(vals, min_val, max_val)
            return np.round(vals, 2).tolist()
        else:
            return np.round(np.random.uniform(min_val, max_val, size), 2).tolist()

    def _generate_varchar_column(self, size: int, spec: Dict) -> List[str]:
        length = spec.get("length", 10)
        cardinality = spec.get("cardinality")
        values = spec.get("values")
        prefix = spec.get("prefix", "")

        if values is not None:
            unique_vals = values
        elif cardinality is not None:
            if prefix:
                unique_vals = [f"{prefix}{i}" for i in range(cardinality)]
            else:
                unique_vals = [self.fake.word()[:length] for _ in range(cardinality)]
        else:
            return [self.fake.pystr(min_chars=length, max_chars=length) for _ in range(size)]

        return [random.choice(unique_vals) for _ in range(size)]

    def _generate_date_column(self, size: int, spec: Dict) -> List[str]:
        from datetime import datetime, timedelta
        start_str = spec.get("start", "2020-01-01")
        end_str = spec.get("end", "2024-12-31")
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        delta = (end - start).days
        days = np.random.randint(0, delta + 1, size)
        dates = [start + timedelta(days=int(d)) for d in days]
        return [d.strftime("%Y-%m-%d") for d in dates]

    def _generate_timestamp_column(self, size: int, spec: Dict) -> List[str]:
        from datetime import datetime, timedelta
        start_str = spec.get("start", "2020-01-01 00:00:00")
        end_str = spec.get("end", "2024-12-31 23:59:59")
        fmt = "%Y-%m-%d %H:%M:%S"
        start = datetime.strptime(start_str, fmt)
        end = datetime.strptime(end_str, fmt)
        delta_seconds = (end - start).total_seconds()
        seconds = np.random.uniform(0, delta_seconds, size)
        timestamps = [start + timedelta(seconds=s) for s in seconds]
        return [t.strftime(fmt) for t in timestamps]

    # ------------------------------------------------------------------
    # DuckDB utilities
    # ------------------------------------------------------------------
    def save_to_duckdb(self, df: pd.DataFrame, db_path: str, table_name: str):
        """Save a pandas DataFrame to a DuckDB table."""
        conn = duckdb.connect(db_path)
        # Register the DataFrame as a temporary view
        conn.register("temp_df", df)
        # Create table from the registered view
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
        conn.unregister("temp_df")
        conn.close()