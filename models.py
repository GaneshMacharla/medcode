# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the MedCodeRL Environment.

Medical Coding & Billing Compliance environment where agents must assign
ICD-10/CPT codes, make billing compliance decisions, and identify fraud patterns.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MedAction(Action):
    """Action for the MedCodeRL environment — a complete medical coding assessment."""

    diagnosis_codes: List[str] = Field(
        ..., min_length=1, max_length=5,
        description="ICD-10-CM diagnosis codes (primary + secondary)"
    )
    procedure_codes: List[str] = Field(
        default_factory=list, max_length=5,
        description="CPT procedure codes"
    )
    decision: Literal["approve", "reject", "review"] = Field(
        ..., description="Billing compliance decision"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Agent confidence in its coding decision"
    )
    reasoning: str = Field(
        ..., min_length=15, max_length=500,
        description="Clinical justification for the coding decision"
    )
    modifier_codes: List[str] = Field(
        default_factory=list, max_length=3,
        description="Optional CPT modifier codes"
    )
    risk_flags: List[str] = Field(
        default_factory=list, max_length=5,
        description="Compliance risk flags identified"
    )


class MedObservation(Observation):
    """Observation from the MedCodeRL environment — a clinical case to code."""

    case_id: str = Field(default="", description="Unique case identifier")
    difficulty: str = Field(default="easy", description="easy | medium | hard")
    clinical_note: str = Field(default="", description="Clinical documentation")
    symptoms: List[str] = Field(default_factory=list, description="Reported symptoms")
    treatments: List[str] = Field(default_factory=list, description="Treatments administered or planned")
    insurance_type: str = Field(default="Private", description="Medicare | Medicaid | Private | Uninsured")
    prior_auth_required: bool = Field(default=False, description="Prior authorization needed")
    treatment_cost: str = Field(default="low", description="low | medium | high")
    patient_age: int = Field(default=0, description="Patient age in years")
    patient_sex: str = Field(default="M", description="M | F")
    provider_specialty: str = Field(default="", description="Treating provider specialty")
    visit_type: str = Field(default="outpatient", description="inpatient | outpatient | emergency | telehealth")
    comorbidities: List[str] = Field(default_factory=list, description="Pre-existing conditions")
    lab_results: Optional[str] = Field(default=None, description="Relevant lab results")
    medications: List[str] = Field(default_factory=list, description="Current medications")

    # Reward breakdown returned after step
    reward_breakdown: Optional[Dict] = Field(default=None, description="Detailed reward breakdown after grading")
    feedback: str = Field(default="", description="Grader feedback text")
