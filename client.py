# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Queryscaler Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import QueryscalerAction, QueryscalerObservation


class QueryscalerEnv(EnvClient[QueryscalerAction, QueryscalerObservation, State]):
    """
    Client for the Queryscaler Environment.

    Example:
        >>> with QueryscalerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description)
        ...
        ...     action = QueryscalerAction(action_type="execute", sql="DESCRIBE orders")
        ...     result = client.step(action)
        ...     print(result.observation.reward)
    """

    def _step_payload(self, action: QueryscalerAction) -> Dict[str, Any]:
        """
        Convert QueryscalerAction to JSON payload for step message.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[QueryscalerObservation]:
        """
        Parse server response into StepResult[QueryscalerObservation].
        """
        obs_data = payload.get("observation", {})
        observation = QueryscalerObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
