# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class QueryscalerAction(Action):
    action_type: str = Field(
        ...,
        description="One of: execute (run any SQL), explain (get query plan), finish (end episode)"
    )
    sql: str | None = Field(None, description="SQL statement to execute or explain")
    reasoning: str | None = Field(None, description="Why this action was chosen")

class QueryscalerObservation(Observation):
    step_number: int = Field(..., description="Current step count")
    task_id: str = Field(..., description="ID of the current task (easy_index, medium_rewrite, hard_workload)")
    task_description: str = Field(..., description="Detailed description of the current task")
    tables: list[str] = Field(default_factory=list, description="Tables present in the database")
    last_result: str | None = Field(None, description="Output from the last SQL command or EXPLAIN")
    last_exit_code: int | None = Field(None, description="0 if last command succeeded, non‑zero otherwise")
    progress: float = Field(0.0, description="Current score (0.0–1.0), proportional to cost reduction")
    hints: list[str] = Field(default_factory=list, description="Helpful suggestions for the agent")
