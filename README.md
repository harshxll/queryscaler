# 🚀 QueryScaler: RL-Based SQL Query Optimization Environment

QueryScaler is a **Reinforcement Learning environment for SQL query optimization**, where an agent learns to improve query performance through iterative actions such as indexing, query rewriting, and execution plan analysis.

Built using **OpenEnv + DuckDB + LLM-based agents**, QueryScaler simulates real-world database optimization tasks with structured rewards and multi-step decision making.

---

## 📌 Motivation

Modern query optimizers:

* Use static heuristics
* Struggle with dynamic workloads
* Cannot adapt to evolving data patterns

👉 QueryScaler introduces:

> **A learning-based optimizer that improves through interaction**

---

## 🧠 Core Idea

We model SQL optimization as a **Markov Decision Process (MDP)**:

| RL Component | QueryScaler Mapping                            |
| ------------ | ---------------------------------------------- |
| State        | Current database + query + history             |
| Action       | SQL execution (index, rewrite, explain)        |
| Reward       | Performance improvement + optimization quality |
| Policy       | Agent strategy (LLM / RL)                      |
| Environment  | DuckDB-backed simulation                       |

---

## 🏗️ Architecture

```
LLM Agent (Policy)
        ↓
Action (SQL / Explain / Finish)
        ↓
QueryScaler Environment
        ↓
DuckDB Execution + Cost Measurement
        ↓
Reward Computation
        ↓
Next State (Observation)
```

---

## ⚙️ Environment Design

### 📂 Implementation

* Environment: 
* Inference Runner: 

---

### 🧾 State (Observation)

Each step returns:

* Task description
* Available tables
* Last query result / error
* Current reward (progress)
* Optimization hints

---

### 🎯 Action Space

Agent outputs JSON:

```json
{"action_type": "execute", "sql": "..."}
{"action_type": "explain", "sql": "..."}
{"action_type": "finish"}
```

---

### 🏆 Reward Function (Key Innovation)

Reward is **multi-component + shaped for learning stability**:

#### 1. Speed Improvement

* Based on runtime vs baseline

#### 2. Structural Quality

* Penalizes:

  * SELECT *
  * excessive joins
* Rewards:

  * filters, limits

#### 3. Strategy Usage

* Index creation
* CTE usage
* Views / subqueries

#### 4. Index Quality

* Matches index columns with query filters

#### 5. Anti-Repetition Penalty

* Prevents redundant actions

---

### 🔥 Final Reward

* Weighted combination of:

  * speed + optimization signals
* Smoothed using EMA for stability
* Encourages **progressive learning**

---

## 🧪 Tasks

### 🟢 Easy: Index Optimization

* Optimize a filtered query on large table

### 🟡 Medium: Query Rewrite + Joins

* Optimize multi-table joins

### 🔴 Hard: Workload Optimization

* Optimize multiple queries
* Constraint: max 2 indexes

---

## ⚡ Key Features

* 🔄 Multi-step optimization environment
* 📊 Real execution-based rewards (DuckDB)
* 🧠 LLM-driven policy (ReAct style)
* 🎯 Reward shaping for convergence
* 🚫 Anti-repetition safeguards
* 📈 Progressive reward scaling

---

## 🤖 LLM Agent Design

From :

### Key Behaviors

* Avoid repeated actions
* Prefer:

  * CREATE INDEX
  * Query rewrites
* Use EXPLAIN only once
* Learn from reward deltas

---

## 🛠️ Tech Stack

* Python
* DuckDB
* OpenEnv framework
* sqlglot (SQL parsing)
* OpenAI / Groq LLM APIs

---

## ▶️ Setup

```bash
git clone <repo>
cd queryscaler
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

```bash
export GROQ_API_KEY=your_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama3-70b-8192
```

---

## ▶️ Run

```bash
python inference_runner.py
```

---

## 📊 Example Workflow

1. Agent inspects schema
2. Runs EXPLAIN once
3. Creates index
4. Rewrites query
5. Observes reward increase
6. Iterates until optimal

---

## ⚠️ Challenges Addressed

* Sparse rewards → solved via shaping
* Large action space → constrained via SQL interface
* Exploration vs exploitation → guided prompt + penalties
* Realistic cost → execution-based measurement

---

## 🌍 Real-World Applications

* Autonomous database tuning
* Cloud query optimization (BigQuery, Snowflake)
* AI-driven compilers
* Data warehouse performance tuning

---

## 🚀 Future Work

* Learned cost model (replace DuckDB estimates)
* Graph-based state encoding (GNNs)
* Multi-agent optimization
* Integration with production DBs

---

## 👨‍💻 Author

**Harshul Arora**
Punjab Engineering College

---

## 🏆 Why This Project Stands Out

✔ Combines RL + Systems + LLMs
✔ Real execution-based feedback (not simulated heuristics)
✔ Generalizable optimization framework
✔ Designed for **agent learning, not rule-based tuning**

---

## 📜 License

MIT License

