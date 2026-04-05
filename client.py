# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MedCodeRL Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MedAction, MedObservation


class MedCodeEnv(
    EnvClient[MedAction, MedObservation, State]
):
    """
    Client for the MedCodeRL Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with MedCodeEnv(base_url="http://localhost:7680") as client:
        ...     result = client.reset()
        ...     print(result.observation.clinical_note)
        ...
        ...     action = MedAction(
        ...         diagnosis_codes=["J02.9"],
        ...         procedure_codes=["99213"],
        ...         decision="approve",
        ...         confidence=0.9,
        ...         reasoning="Acute pharyngitis with appropriate E&M coding.",
        ...         risk_flags=[]
        ...     )
        ...     result = client.step(action)
        ...     print(f"Score: {result.reward}")
    """

    def _step_payload(self, action: MedAction) -> Dict:
        """
        Convert MedAction to JSON payload for step message.

        Args:
            action: MedAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "diagnosis_codes": action.diagnosis_codes,
            "procedure_codes": action.procedure_codes,
            "decision": action.decision,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "modifier_codes": action.modifier_codes,
            "risk_flags": action.risk_flags,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MedObservation]:
        """
        Parse server response into StepResult[MedObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MedObservation
        """
        obs_data = payload.get("observation", {})
        observation = MedObservation(
            case_id=obs_data.get("case_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            clinical_note=obs_data.get("clinical_note", ""),
            symptoms=obs_data.get("symptoms", []),
            treatments=obs_data.get("treatments", []),
            insurance_type=obs_data.get("insurance_type", "Private"),
            prior_auth_required=obs_data.get("prior_auth_required", False),
            treatment_cost=obs_data.get("treatment_cost", "low"),
            patient_age=obs_data.get("patient_age", 0),
            patient_sex=obs_data.get("patient_sex", "M"),
            provider_specialty=obs_data.get("provider_specialty", ""),
            visit_type=obs_data.get("visit_type", "outpatient"),
            comorbidities=obs_data.get("comorbidities", []),
            lab_results=obs_data.get("lab_results"),
            medications=obs_data.get("medications", []),
            reward_breakdown=obs_data.get("reward_breakdown"),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
