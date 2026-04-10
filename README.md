# QueryScaler

### Reinforcement Learning Environment for SQL Query Optimization

QueryScaler is a learning-based SQL query optimization system that models query tuning as a Reinforcement Learning (RL) problem. An agent iteratively interacts with a database environment, applying transformations such as indexing and query rewriting, and receives feedback based on actual execution performance.

The system integrates RL principles, a real execution engine (DuckDB), and LLM-based policies to enable adaptive and data-driven query optimization.

---

## Problem Statement

Traditional database query optimizers rely on:

* Static heuristics
* Cost estimation models
* Limited adaptability to evolving workloads

These approaches often fail to generalize across diverse queries and dynamic data distributions.

QueryScaler addresses this limitation by introducing an environment where optimization strategies are learned through interaction and reward feedback.

---

## Core Idea

SQL query optimization is modeled as a Markov Decision Process (MDP):

| RL Concept  | QueryScaler Mapping                              |
| ----------- | ------------------------------------------------ |
| State       | Database state, query, and interaction history   |
| Action      | SQL execution, explain plan, or termination      |
| Reward      | Performance improvement and optimization quality |
| Policy      | Agent (LLM or RL-based)                          |
| Environment | DuckDB-backed execution system                   |

---

## Project Structure

```bash id="structure_final"
queryscaler/
├── client.py                      # Client interface for interacting with the environment
├── inference.py                   # LLM-based agent runner (policy execution)
├── main.py                        # Entry point / orchestration script
├── models.py                      # Action and Observation schemas
├── table_generator.py             # Synthetic data generation utilities
│
├── server/
│   ├── app.py                     # OpenEnv server entry point
│   ├── queryscaler_environment.py # Core RL environment implementation
│   └── __init__.py
│
├── test_env.py                    # Environment tests
├── test_env_llm.py                # LLM interaction tests
├── test_reward_environment.py     # Reward function validation
├── test_table_generator.py        # Data generation tests
│
├── Dockerfile                     # Container configuration
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
├── openenv.yaml                   # OpenEnv configuration
├── prevalidate.sh                 # Pre-submission validation script
├── uv.lock                        # Dependency lock file
├── README.md                      # Documentation
└── __init__.py
```

> Note: Automatically generated folders such as `__pycache__/` and compiled files (`*.pyc`) are excluded from version control and documentation.

---

## Environment Design

### Core Environment

Implemented in: 

The environment simulates a query optimization workflow by:

* Maintaining database state
* Executing SQL commands
* Tracking optimization progress
* Providing structured observations to the agent

---

## Action and Observation Interface

### Action Schema

```python id="action_schema"
class QueryscalerAction(Action):
    action_type: str   # "execute", "explain", or "finish"
    sql: Optional[str]
    reasoning: Optional[str]
```

### Observation Schema

```python id="observation_schema"
class QueryscalerObservation(Observation):
    step_number: int
    task_id: str
    task_description: str
    tables: list[str]
    last_result: Optional[str]
    last_exit_code: Optional[int]
    progress: float
    hints: list[str]
```

This structured interface enables:

* Compatibility with OpenEnv-style environments
* Seamless integration with LLM agents
* Multi-step reasoning and decision-making workflows

---

## Action Types

| Action Type | Description                                         |
| ----------- | --------------------------------------------------- |
| execute     | Execute SQL statements (e.g., CREATE INDEX, SELECT) |
| explain     | Retrieve query execution plans                      |
| finish      | Terminate the optimization episode                  |

---

## Reward System

QueryScaler employs a composite reward function designed to encourage meaningful optimization behavior.

### Components

1. **Performance Improvement**

   * Based on execution time relative to a baseline query

2. **Structural Quality**

   * Penalizes inefficient constructs such as `SELECT *` and excessive joins
   * Rewards use of filters and limits

3. **Strategy Utilization**

   * Rewards actions such as index creation, CTE usage, views, and schema modifications

4. **Index Effectiveness**

   * Measures alignment between index columns and query predicates

5. **Repetition Penalty**

   * Penalizes redundant or repeated actions

---

### Final Reward

```id="reward_formula"
reward = 0.35 * speed_score + 0.65 * optimization_signal
```

Additional mechanisms:

* Exponential Moving Average (EMA) smoothing
* Progressive reward scaling for stable convergence

---

## Tasks

### Easy: Index Optimization

* Optimize a single-table query using indexing

### Medium: Query Rewrite and Join Optimization

* Improve performance of multi-table joins

### Hard: Workload Optimization

* Optimize multiple queries with constraints (e.g., limited number of indexes)

---

## Data Generation

Implemented in: 

Features include:

* Synthetic dataset generation
* Support for multiple statistical distributions (uniform, Zipf, log-normal)
* Configurable schema and null values
* Scalable workloads for benchmarking

---

## Agent Design

Implemented in: 

Key characteristics:

* ReAct-style interaction loop
* Reward-aware decision making
* Prevention of repeated actions
* Strategy-oriented prompting

---

## Typical Optimization Workflow

1. Inspect schema using `DESCRIBE`
2. Analyze query plan using `EXPLAIN`
3. Apply index creation
4. Rewrite queries for efficiency
5. Iterate based on reward feedback
6. Terminate optimization

---

## Setup

```bash id="setup_final"
git clone <repository_url>
cd queryscaler
pip install -r requirements.txt
```

---

## Environment Configuration

```bash id="env_final"
export GROQ_API_KEY=your_api_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama3-70b-8192
```

---

## Running the System

```bash id="run_final"
python inference.py
```

---

## Testing

```bash id="test_final"
pytest
```

---

## Key Features

* Multi-step reinforcement learning environment
* Execution-based reward computation using DuckDB
* Integration with LLM-based agents
* Structured action-observation interface
* Robust reward shaping for stable learning

---

## Challenges Addressed

| Challenge               | Approach                               |
| ----------------------- | -------------------------------------- |
| Sparse rewards          | Reward shaping and progressive scaling |
| Large action space      | Structured SQL-based action design     |
| Instability in learning | EMA smoothing                          |
| Agent repetition        | Explicit penalties and safeguards      |

---

## Applications

* Autonomous database tuning systems
* Query optimization in analytical engines
* AI-assisted compilers and planners
* Performance optimization in data platforms

---

## Future Work

* Learned cost models using machine learning
* Graph-based representations of query plans
* Multi-agent optimization frameworks
* Integration with production-grade databases

---

## Author

Harshul Arora
Punjab Engineering College
