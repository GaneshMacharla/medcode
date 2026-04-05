"""
MedCodeRL - Baseline Inference Script

Uses the OpenAI API client to run a model against the MedCodeRL environment.
Reads API credentials from environment variables.

Usage:
    export HF_TOKEN="your-key"
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    python inference.py
"""

import json
import os
import re
import sys
import time
from typing import Optional

from openai import OpenAI

# Add project root to path so we can import the environment directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.my_env_environment import MyEnvironment, _load_task_cases
from models import MedAction


# ----- Configuration -----

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable (Your Hugging Face / API key).")
    sys.exit(1)

# Number of cases to evaluate per difficulty level
CASES_PER_DIFFICULTY = int(os.environ.get("CASES_PER_DIFFICULTY", "5"))
MAX_RETRIES = 2


def create_client() -> OpenAI:
    """Create an OpenAI-compatible client."""
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def extract_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON from an LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return None


SYSTEM_PROMPT = """You are an expert medical coding and billing compliance specialist.
Your role is to:
1. Analyze clinical documentation
2. Assign appropriate ICD-10-CM diagnosis codes
3. Assign appropriate CPT procedure codes
4. Make billing compliance decisions (approve, reject, or flag for review)
5. Identify compliance risk flags
6. Provide clinical reasoning for your decisions

You have deep knowledge of:
- ICD-10-CM coding guidelines and conventions
- CPT coding and modifier usage
- Medicare/Medicaid billing rules
- Medical necessity requirements
- Common compliance violations (upcoding, unbundling, fraudulent billing)
- Clinical documentation integrity
- Prior authorization requirements

Always respond with ONLY a valid JSON object in the exact format requested.
Be precise with your ICD-10 and CPT codes.
Consider the clinical documentation, symptoms, treatments, and insurance type when making decisions.
Flag any compliance concerns in risk_flags."""


def format_observation(obs) -> str:
    """Format an observation into a readable prompt for the LLM."""
    parts = [
        f"## Medical Coding Case: {obs.case_id}",
        f"**Difficulty:** {obs.difficulty}",
        f"**Visit Type:** {obs.visit_type}",
        f"**Provider Specialty:** {obs.provider_specialty}",
        "",
        "### Patient Information",
        f"- **Age:** {obs.patient_age} | **Sex:** {obs.patient_sex}",
        f"- **Insurance:** {obs.insurance_type}",
        f"- **Prior Authorization Required:** {'Yes' if obs.prior_auth_required else 'No'}",
        f"- **Treatment Cost Tier:** {obs.treatment_cost}",
        "",
        "### Clinical Note",
        obs.clinical_note,
        "",
        "### Symptoms",
        ", ".join(obs.symptoms) if obs.symptoms else "None reported",
        "",
        "### Treatments",
        ", ".join(obs.treatments) if obs.treatments else "None",
    ]

    if obs.comorbidities:
        parts += ["", "### Comorbidities", ", ".join(obs.comorbidities)]
    if obs.lab_results:
        parts += ["", "### Lab Results", obs.lab_results]
    if obs.medications:
        parts += ["", "### Current Medications", ", ".join(obs.medications)]

    return "\n".join(parts)


ACTION_PROMPT = """
Based on the clinical case above, provide your medical coding and billing compliance assessment.

You MUST respond with a valid JSON object containing exactly these fields:

{
    "diagnosis_codes": ["<ICD-10 code(s)>"],
    "procedure_codes": ["<CPT code(s) if applicable, or empty list>"],
    "decision": "<approve|reject|review>",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<15-500 character clinical justification>",
    "modifier_codes": ["<optional CPT modifiers, or empty list>"],
    "risk_flags": ["<compliance risk flags identified, or empty list>"]
}

Guidelines:
- Use standard ICD-10-CM codes (e.g., J06.9 for upper respiratory infection)
- Use standard CPT codes (5 digits, e.g., 99213 for office visit)
- decision: "approve" if coding is appropriate, "reject" if non-compliant, "review" if ambiguous
- confidence: your certainty (0.0 = unsure, 1.0 = certain)
- reasoning: explain WHY you chose these codes and this decision
- risk_flags: compliance risks (e.g., "upcoding_risk", "missing_documentation", "bundling_violation")

IMPORTANT: Respond ONLY with the JSON object, no additional text.
"""


def call_llm(client: OpenAI, obs) -> Optional[dict]:
    """Call the LLM to get a coding decision for a clinical case."""
    formatted = format_observation(obs)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": formatted + "\n\n" + ACTION_PROMPT},
                ],
                temperature=0.1,
                max_tokens=800,
            )

            content = response.choices[0].message.content
            if not content:
                print(f"  [Attempt {attempt+1}] Empty response from LLM")
                continue

            action = extract_json_from_response(content)
            if action is None:
                print(f"  [Attempt {attempt+1}] Failed to parse JSON from response")
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                continue

            # Sanitize fields
            if "diagnosis_codes" not in action or not isinstance(action["diagnosis_codes"], list):
                action["diagnosis_codes"] = [action["diagnosis_codes"]] if isinstance(action.get("diagnosis_codes"), str) else ["R69"]
            if "procedure_codes" not in action:
                action["procedure_codes"] = []
            if isinstance(action.get("procedure_codes"), str):
                action["procedure_codes"] = [action["procedure_codes"]]
            if "decision" not in action:
                action["decision"] = "review"
            if "confidence" not in action:
                action["confidence"] = 0.5
            action["confidence"] = max(0.0, min(1.0, float(action["confidence"])))
            if "reasoning" not in action or len(str(action.get("reasoning", ""))) < 15:
                action["reasoning"] = "Medical coding assessment based on clinical documentation review and compliance guidelines."
            if "modifier_codes" not in action:
                action["modifier_codes"] = []
            if "risk_flags" not in action:
                action["risk_flags"] = []

            return action

        except Exception as e:
            print(f"  [Attempt {attempt+1}] API error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    return None


def get_fallback_action() -> dict:
    """Return a safe fallback action if LLM fails."""
    return {
        "diagnosis_codes": ["R69"],
        "procedure_codes": ["99213"],
        "decision": "review",
        "confidence": 0.1,
        "reasoning": "Unable to obtain LLM response. Flagging for manual review as a safety measure.",
        "modifier_codes": [],
        "risk_flags": ["llm_failure"],
    }


def run_evaluation():
    """Run the baseline evaluation across all difficulty levels."""
    print("=" * 70)
    print("MedCodeRL - Baseline Inference Script")
    print("=" * 70)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Cases per difficulty: {CASES_PER_DIFFICULTY}")
    print("=" * 70)

    client = create_client()
    env = MyEnvironment()

    all_scores = []
    results_by_difficulty = {}

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n{'─' * 50}")
        print(f"  Running {difficulty.upper()} tasks")
        print(f"{'─' * 50}")

        difficulty_scores = []
        available = len(env._task_cases.get(difficulty, []))
        num_cases = min(CASES_PER_DIFFICULTY, available)

        if num_cases == 0:
            print(f"  No cases available for {difficulty}")
            continue

        for i in range(num_cases):
            # Reset environment for this difficulty
            obs = env.reset(task_id=difficulty)
            print(f"\n  Case {i+1}/{num_cases}: {obs.case_id}")

            # Get LLM action
            action_dict = call_llm(client, obs)
            if action_dict is None:
                print("    ⚠ LLM failed, using fallback action")
                action_dict = get_fallback_action()

            # Build MedAction
            med_action = MedAction(
                diagnosis_codes=action_dict["diagnosis_codes"],
                procedure_codes=action_dict.get("procedure_codes", []),
                decision=action_dict["decision"],
                confidence=action_dict["confidence"],
                reasoning=action_dict["reasoning"],
                modifier_codes=action_dict.get("modifier_codes", []),
                risk_flags=action_dict.get("risk_flags", []),
            )

            # Step the environment
            try:
                result_obs = env.step(med_action)
                score = result_obs.reward if result_obs.reward is not None else 0.0
                difficulty_scores.append(score)
                all_scores.append(score)

                print(f"    Score: {score:.4f}")
                print(f"    Decision: {action_dict.get('decision', 'N/A')}")
                print(f"    Diagnosis: {action_dict.get('diagnosis_codes', [])}")
                print(f"    Procedure: {action_dict.get('procedure_codes', [])}")

                if result_obs.reward_breakdown:
                    gc = result_obs.reward_breakdown.get("grade_components", {})
                    if gc:
                        print(f"    Components: diag={gc.get('diagnosis_accuracy', 0):.2f} "
                              f"proc={gc.get('procedure_accuracy', 0):.2f} "
                              f"dec={gc.get('decision_accuracy', 0):.2f}")
                    pens = result_obs.reward_breakdown.get("penalties", {})
                    if pens:
                        print(f"    Penalties: {list(pens.keys())}")

                if result_obs.feedback:
                    print(f"    Feedback: {result_obs.feedback}")

            except Exception as e:
                print(f"    ✗ Step failed: {e}")
                difficulty_scores.append(0.0)
                all_scores.append(0.0)

            # Rate limiting
            time.sleep(0.5)

        if difficulty_scores:
            avg = sum(difficulty_scores) / len(difficulty_scores)
            results_by_difficulty[difficulty] = {
                "scores": difficulty_scores,
                "average": round(avg, 4),
                "count": len(difficulty_scores),
            }
            print(f"\n  {difficulty.upper()} Average: {avg:.4f} ({len(difficulty_scores)} cases)")

    # Final summary
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS")
    print(f"{'=' * 70}")

    for diff, result in results_by_difficulty.items():
        print(f"  {diff.upper():>8}: {result['average']:.4f}  ({result['count']} cases)")

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"\n  {'OVERALL':>8}: {overall:.4f}  ({len(all_scores)} total cases)")
    else:
        overall = 0.0
        print("\n  No scores recorded.")

    print(f"{'=' * 70}")

    # Write results to file
    results_output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "cases_per_difficulty": CASES_PER_DIFFICULTY,
        "results_by_difficulty": results_by_difficulty,
        "overall_score": round(overall, 4),
        "total_cases": len(all_scores),
    }

    with open("baseline_results.json", "w") as f:
        json.dump(results_output, f, indent=2)
    print(f"\nResults saved to baseline_results.json")

    return overall


if __name__ == "__main__":
    score = run_evaluation()
    sys.exit(0 if score > 0 else 1)
