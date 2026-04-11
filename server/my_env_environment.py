# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MedCodeRL Environment Implementation.

Medical Coding & Billing Compliance environment where agents must:
1. Assign correct ICD-10 diagnosis codes
2. Assign correct CPT procedure codes
3. Make billing compliance decisions (approve/reject/review)
4. Provide clinical reasoning
5. Identify compliance risks

Contains: environment logic, deterministic grader, shaped rewards, action validation.
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MedAction, MedObservation
except ImportError:
    from models import MedAction, MedObservation


# ──────────────────────────────────────────────
#  Task loader
# ──────────────────────────────────────────────

TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")


def _load_task_cases(difficulty: str) -> List[Dict[str, Any]]:
    """Load clinical cases for a given difficulty level."""
    filepath = os.path.join(TASKS_DIR, f"{difficulty}.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


# ──────────────────────────────────────────────
#  Action validation
# ──────────────────────────────────────────────

ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$", re.IGNORECASE)
CPT_PATTERN = re.compile(r"^\d{5}$")
HCPCS_PATTERN = re.compile(r"^[A-Z]\d{4}$", re.IGNORECASE)
SCORE_EPSILON = 0.01


def _to_open_interval_score(value: float) -> float:
    """Map a score to the strict open interval (0, 1).

    Guarantees the returned value satisfies  0 < value < 1.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0

    if not math.isfinite(score):
        score = 0.0

    # Clamp into the safe open interval (SCORE_EPSILON, 1 - SCORE_EPSILON)
    score = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, score))
    return score


def _rounded_open_interval_score(value: float, ndigits: int = 4) -> float:
    """Round score while preserving strict open interval bounds."""
    clamped = _to_open_interval_score(value)
    rounded = round(clamped, ndigits)
    # Re-clamp after rounding to guarantee strict (0, 1)
    return _to_open_interval_score(rounded)


def _rounded_component_score(value: float, ndigits: int = 4) -> float:
    """Normalize component scores to strict open interval for validator safety."""
    return _rounded_open_interval_score(value, ndigits)


def _validate_action(action_dict: dict) -> Tuple[bool, List[str]]:
    """Validate an action dict. Returns (is_valid, error_list)."""
    errors: List[str] = []

    diag = action_dict.get("diagnosis_codes", [])
    if not isinstance(diag, list) or len(diag) == 0:
        errors.append("At least one diagnosis code is required")
    elif len(diag) > 5:
        errors.append("Maximum 5 diagnosis codes allowed")
    else:
        for code in diag:
            if not ICD10_PATTERN.match(str(code).strip()):
                errors.append(f"Invalid ICD-10 format: {code}")

    proc = action_dict.get("procedure_codes", [])
    if isinstance(proc, list):
        for code in proc:
            c = str(code).strip()
            if not CPT_PATTERN.match(c) and not HCPCS_PATTERN.match(c):
                errors.append(f"Invalid CPT/HCPCS format: {code}")

    decision = action_dict.get("decision", "")
    if str(decision).lower() not in ("approve", "reject", "review"):
        errors.append(f"Invalid decision: {decision}")

    confidence = action_dict.get("confidence", -1)
    try:
        conf_val = float(confidence)
        if conf_val < 0.0 or conf_val > 1.0:
            errors.append("confidence must be 0.0-1.0")
    except (TypeError, ValueError):
        errors.append("confidence must be a number")

    reasoning = action_dict.get("reasoning", "")
    if not isinstance(reasoning, str) or len(reasoning) < 15:
        errors.append("reasoning must be at least 15 characters")

    return len(errors) == 0, errors


# ──────────────────────────────────────────────
#  Deterministic grader
# ──────────────────────────────────────────────

def _set_similarity(predicted: List[str], ground_truth: List[str]) -> float:
    """Jaccard similarity between two code sets."""
    if not ground_truth and not predicted:
        return 1.0
    pred_set: Set[str] = set(c.strip().upper() for c in predicted if c)
    gt_set: Set[str] = set(c.strip().upper() for c in ground_truth if c)
    if not gt_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    intersection = pred_set & gt_set
    union = pred_set | gt_set
    return len(intersection) / len(union) if union else 1.0


def _partial_code_match(predicted: List[str], ground_truth: List[str]) -> float:
    """Partial-credit matching for medical codes (prefix similarity)."""
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    pred_list = [c.strip().upper() for c in predicted if c]
    gt_list = [c.strip().upper() for c in ground_truth if c]
    total = 0.0
    for gt_code in gt_list:
        best = 0.0
        for pred_code in pred_list:
            if pred_code == gt_code:
                best = 1.0
                break
            gt_base = gt_code.split(".")[0]
            pred_base = pred_code.split(".")[0]
            if gt_base == pred_base:
                best = max(best, 0.5)
            elif len(gt_base) >= 3 and gt_base[:3] == pred_base[:3]:
                best = max(best, 0.25)
        total += best
    return total / len(gt_list)


def _grade(action_dict: dict, ground_truth: dict) -> Dict[str, float]:
    """
    Deterministic grading — 6 weighted components → score in [0.0, 1.0].

    Weights: diagnosis 35%, procedure 20%, decision 25%, reasoning 10%,
    risk flags 5%, confidence calibration 5%.
    """
    pred_diag = action_dict.get("diagnosis_codes", [])
    gt_diag = ground_truth.get("diagnosis_codes", [])
    pred_proc = action_dict.get("procedure_codes", [])
    gt_proc = ground_truth.get("procedure_codes", [])
    pred_dec = str(action_dict.get("decision", "")).lower()
    gt_dec = str(ground_truth.get("decision", "")).lower()
    pred_reasoning = str(action_dict.get("reasoning", ""))
    pred_risk = action_dict.get("risk_flags", [])
    gt_risk = ground_truth.get("risk_flags", [])
    pred_conf = float(action_dict.get("confidence", 0.5))

    # 1. Diagnosis codes (35%)
    diag_exact = _set_similarity(pred_diag, gt_diag)
    diag_partial = _partial_code_match(pred_diag, gt_diag)
    diag_score = 0.6 * diag_exact + 0.4 * diag_partial

    # 2. Procedure codes (20%)
    proc_exact = _set_similarity(pred_proc, gt_proc)
    proc_partial = _partial_code_match(pred_proc, gt_proc)
    proc_score = 0.6 * proc_exact + 0.4 * proc_partial

    # 3. Decision (25%)
    if pred_dec == gt_dec:
        dec_score = 1.0
    elif pred_dec == "review" and gt_dec in ("approve", "reject"):
        dec_score = 0.3
    elif pred_dec in ("approve", "reject") and gt_dec == "review":
        dec_score = 0.2
    else:
        dec_score = 0.0

    # 4. Reasoning quality (10%)
    reasoning_lower = pred_reasoning.lower()
    r_score = 0.0
    if len(pred_reasoning) >= 20:
        r_score += 0.3
    if len(pred_reasoning) >= 50:
        r_score += 0.2
    med_terms = [
        "icd", "cpt", "diagnosis", "procedure", "coding", "compliance",
        "medical", "clinical", "treatment", "patient", "billing",
        "authorization", "insurance", "modifier", "documentation",
        "justified", "appropriate", "medically necessary", "guideline",
    ]
    r_score += min(0.5, sum(1 for t in med_terms if t in reasoning_lower) * 0.1)
    r_score = min(1.0, r_score)

    # 5. Risk flags (5%)
    risk_score = _set_similarity(pred_risk, gt_risk)

    # 6. Confidence calibration (5%)
    correctness = diag_score * 0.5 + proc_score * 0.3 + dec_score * 0.2
    conf_score = max(0.0, 1.0 - abs(pred_conf - correctness) * 2.0)

    total = (
        diag_score * 0.35
        + proc_score * 0.20
        + dec_score * 0.25
        + r_score * 0.10
        + risk_score * 0.05
        + conf_score * 0.05
    )

    return {
        "score": _rounded_open_interval_score(total, 4),
        "diagnosis_accuracy": _rounded_component_score(diag_score, 4),
        "procedure_accuracy": _rounded_component_score(proc_score, 4),
        "decision_accuracy": _rounded_component_score(dec_score, 4),
        "reasoning_quality": _rounded_component_score(r_score, 4),
        "risk_identification": _rounded_component_score(risk_score, 4),
        "confidence_calibration": _rounded_component_score(conf_score, 4),
    }


# ──────────────────────────────────────────────
#  Shaped reward engine
# ──────────────────────────────────────────────

def _compute_reward(action_dict: dict, ground_truth: dict, difficulty: str = "easy") -> Dict:
    """
    Shaped reward = base_grade + bonuses − penalties.

    Penalties: upcoding, undercoding, wrong denial/approval, unnecessary procedure,
    missing primary code, low confidence.
    Bonuses: perfect diagnosis, good reasoning, all risk flags.
    """
    grade_result = _grade(action_dict, ground_truth)
    base = grade_result["score"]

    pred_diag = set(c.strip().upper() for c in action_dict.get("diagnosis_codes", []) if c)
    gt_diag = set(c.strip().upper() for c in ground_truth.get("diagnosis_codes", []) if c)
    gt_diag_list = [c.strip().upper() for c in ground_truth.get("diagnosis_codes", []) if c]
    pred_proc = set(c.strip().upper() for c in action_dict.get("procedure_codes", []) if c)
    gt_proc = set(c.strip().upper() for c in ground_truth.get("procedure_codes", []) if c)
    pred_dec = str(action_dict.get("decision", "")).lower()
    gt_dec = str(ground_truth.get("decision", "")).lower()
    pred_conf = float(action_dict.get("confidence", 0.5))

    penalties: Dict[str, float] = {}
    bonuses: Dict[str, float] = {}

    # Penalties
    if len(pred_proc) > len(gt_proc) + 1:
        penalties["upcoding"] = -0.15
    if gt_diag and len(pred_diag & gt_diag) < len(gt_diag) * 0.5:
        penalties["undercoding"] = -0.10
    if pred_dec == "reject" and gt_dec == "approve":
        penalties["wrong_denial"] = -0.20
    if pred_dec == "approve" and gt_dec == "reject":
        penalties["wrong_approval"] = -0.25
    if pred_proc - gt_proc:
        penalties["unnecessary_procedure"] = -0.10
    if gt_diag_list and gt_diag_list[0] not in pred_diag:
        penalties["missing_primary_code"] = -0.15
    if pred_conf < 0.2:
        penalties["low_confidence"] = -0.05

    # Bonuses
    if grade_result["diagnosis_accuracy"] >= 0.99:
        bonuses["perfect_diagnosis"] = 0.05
    if grade_result["reasoning_quality"] >= 0.8:
        bonuses["good_reasoning"] = 0.03
    if grade_result["risk_identification"] >= 0.99:
        bonuses["all_risk_flags"] = 0.05

    diff_mult = {"easy": 0.8, "medium": 1.0, "hard": 1.2}.get(difficulty, 1.0)
    total_penalty = sum(penalties.values()) * diff_mult
    total_bonus = sum(bonuses.values())
    final = _to_open_interval_score(base + total_penalty + total_bonus)

    feedback_parts = []
    if penalties:
        feedback_parts.append(f"Penalties: {', '.join(penalties.keys())}")
    if bonuses:
        feedback_parts.append(f"Bonuses: {', '.join(bonuses.keys())}")
    if not penalties and not bonuses:
        feedback_parts.append("Clean submission.")

    return {
        "score": _rounded_open_interval_score(final, 4),
        "breakdown": {
            "base_grade": round(base, 4),
            "total_penalty": round(total_penalty, 4),
            "total_bonus": round(total_bonus, 4),
            "penalties": {k: round(v, 4) for k, v in penalties.items()},
            "bonuses": {k: round(v, 4) for k, v in bonuses.items()},
            "grade_components": grade_result,
        },
        "feedback": " | ".join(feedback_parts),
    }


# ──────────────────────────────────────────────
#  Core Environment
# ──────────────────────────────────────────────

class MyEnvironment(Environment):
    """
    MedCodeRL — Medical Coding & Billing Compliance Environment.

    90 realistic clinical cases (30 easy, 30 medium, 30 hard) covering:
    - Straightforward coding (easy)
    - Multi-diagnosis with comorbidities (medium)
    - Compliance dilemmas: upcoding, unbundling, fraud, ethical edge cases (hard)

    OpenEnv-compliant: reset() / step() / state property.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the MedCodeRL environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        # Load all task cases
        self._task_cases: Dict[str, List[Dict]] = {}
        self._case_index: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        for diff in ("easy", "medium", "hard"):
            self._task_cases[diff] = _load_task_cases(diff)

        self._current_case: Optional[Dict] = None
        self._current_difficulty: str = "easy"
        self._done: bool = True
        self._action_history: List[dict] = []

    def _pick_case(self, task_id: Optional[str] = None) -> Dict:
        """Select a case by difficulty or specific case_id."""
        if task_id in ("easy", "medium", "hard"):
            difficulty = task_id
        elif task_id:
            # Search for specific case_id
            for diff in ("easy", "medium", "hard"):
                for case in self._task_cases.get(diff, []):
                    if case["id"] == task_id:
                        self._current_difficulty = diff
                        return case
            raise ValueError(f"Case not found: {task_id}")
        else:
            # Cycle through difficulties
            total = sum(self._case_index.values())
            difficulty = ["easy", "medium", "hard"][total % 3]

        cases = self._task_cases.get(difficulty, [])
        if not cases:
            raise ValueError(f"No cases for difficulty: {difficulty}")
        idx = self._case_index[difficulty] % len(cases)
        self._case_index[difficulty] = idx + 1
        self._current_difficulty = difficulty
        return cases[idx]

    def _build_observation(self, case: Dict, done: bool = False,
                           reward: Optional[float] = None,
                           reward_breakdown: Optional[Dict] = None,
                           feedback: str = "") -> MedObservation:
        """Build a MedObservation from a case dict."""
        inp = case.get("input", case)
        return MedObservation(
            case_id=case.get("id", ""),
            difficulty=case.get("difficulty", self._current_difficulty),
            clinical_note=inp.get("clinical_note", ""),
            symptoms=inp.get("symptoms", []),
            treatments=inp.get("treatments", []),
            insurance_type=inp.get("insurance_type", "Private"),
            prior_auth_required=inp.get("prior_auth_required", False),
            treatment_cost=inp.get("treatment_cost", "low"),
            patient_age=inp.get("patient_age", 0),
            patient_sex=inp.get("patient_sex", "M"),
            provider_specialty=inp.get("provider_specialty", ""),
            visit_type=inp.get("visit_type", "outpatient"),
            comorbidities=inp.get("comorbidities", []),
            lab_results=inp.get("lab_results"),
            medications=inp.get("medications", []),
            reward_breakdown=reward_breakdown,
            feedback=feedback,
            done=done,
            reward=reward,
            metadata={
                "step_count": self._state.step_count,
                "difficulty": self._current_difficulty,
            },
        )

    def reset(self, task_id: Optional[str] = None, **kwargs) -> MedObservation:
        """
        Reset the environment to a new episode.

        Args:
            task_id: 'easy', 'medium', 'hard', or a specific case_id.

        Returns:
            MedObservation with the clinical case to code.
        """
        # Accept task_id from kwargs if not provided directly
        if task_id is None:
            task_id = kwargs.get("task_id")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._done = False
        self._action_history = []

        self._current_case = self._pick_case(task_id)

        return self._build_observation(
            self._current_case,
            done=False,
            reward=_to_open_interval_score(0.0),
        )

    def step(self, action: MedAction) -> MedObservation:  # type: ignore[override]
        """
        Execute a step: grade the agent's medical coding action.

        Args:
            action: MedAction with diagnosis codes, procedure codes, decision, etc.

        Returns:
            MedObservation with reward and grading breakdown.
        """
        if self._done:
            return self._build_observation(
                self._current_case or {},
                done=True,
                reward=_to_open_interval_score(0.0),
                feedback="Episode already done. Call reset().",
            )

        self._state.step_count += 1

        # Convert action to dict
        action_dict = {
            "diagnosis_codes": action.diagnosis_codes,
            "procedure_codes": action.procedure_codes,
            "decision": action.decision,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "modifier_codes": action.modifier_codes,
            "risk_flags": action.risk_flags,
        }

        # Validate
        is_valid, errors = _validate_action(action_dict)

        if not is_valid:
            self._action_history.append({"action": action_dict, "valid": False})
            if self._state.step_count >= 3:
                self._done = True
            return self._build_observation(
                self._current_case or {},
                done=self._done,
                reward=_to_open_interval_score(0.0),
                feedback=f"Invalid action: {'; '.join(errors)}",
            )

        # Grade against ground truth
        ground_truth = self._current_case.get("ground_truth", {})
        reward_result = _compute_reward(action_dict, ground_truth, self._current_difficulty)

        self._action_history.append({"action": action_dict, "valid": True})
        self._done = True  # single-step episode for valid actions

        score = _to_open_interval_score(reward_result["score"])

        return self._build_observation(
            self._current_case or {},
            done=True,
            reward=score,
            reward_breakdown=reward_result.get("breakdown"),
            feedback=reward_result.get("feedback", ""),
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            State with episode_id, step_count, and metadata
        """
        self._state.metadata = {
            "done": self._done,
            "difficulty": self._current_difficulty,
            "case_id": self._current_case.get("id", "") if self._current_case else "",
            "reset_count": self._reset_count,
        }
        return self._state

    def get_task_info(self) -> dict:
        """Return task metadata for discovery endpoints."""
        return {
            "tasks": {
                diff: {
                    "count": len(cases),
                    "description": {
                        "easy": "Straightforward cases with single diagnoses and direct ICD-10/CPT mapping.",
                        "medium": "Multi-diagnosis cases with comorbidities, insurance considerations, and partial ambiguity.",
                        "hard": "Complex compliance dilemmas: upcoding, unbundling, fraud detection, ethical edge cases.",
                    }.get(diff, ""),
                }
                for diff, cases in self._task_cases.items()
            },
            "total_cases": sum(len(c) for c in self._task_cases.values()),
        }
