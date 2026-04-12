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
import math
import os
import re
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from openai import OpenAI

# Add project root to path so we can import the environment directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.my_env_environment import MyEnvironment
from models import MedAction


# ----- Configuration -----

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# Support both HF_TOKEN and OPENAI_API_KEY for maximum compatibility
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Number of cases to evaluate per difficulty level
CASES_PER_DIFFICULTY = int(os.environ.get("CASES_PER_DIFFICULTY", "5"))
MAX_RETRIES = 2

# Global timeout safety (inference must complete in < 20 minutes)
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", "1100"))  # ~18.3 min
_start_time = time.time()
SCORE_EPSILON = 0.01


def to_open_interval_score(value: float) -> float:
    """Map scores to strict open interval (0, 1) for validator compliance.

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


def rounded_open_interval_score(value: float, ndigits: int = 4) -> float:
    """Round score for logs/reports while preserving strict (0, 1) bounds."""
    clamped = to_open_interval_score(value)
    rounded = round(clamped, ndigits)
    # Re-clamp after rounding to guarantee strict (0, 1)
    return to_open_interval_score(rounded)


def is_strict_open_interval(value: float) -> bool:
    """Return True if value is strictly between 0 and 1 and finite."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score) and 0.0 < score < 1.0


class TeeStream:
    """Write output to multiple streams (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _resolve_log_path() -> str:
    """Resolve output log path with timestamped default per run."""
    configured = os.environ.get("LOG_FILE")
    if configured:
        return configured

    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"inference_{timestamp}.log")


def _rotate_log_if_needed(log_path: str):
    """Rotate a fixed log file when it exceeds configured size."""
    max_bytes = int(os.environ.get("LOG_ROTATE_MAX_BYTES", "0"))
    backups = int(os.environ.get("LOG_ROTATE_BACKUPS", "3"))

    if max_bytes <= 0 or backups <= 0:
        return
    if not os.path.exists(log_path):
        return
    if os.path.getsize(log_path) < max_bytes:
        return

    for idx in range(backups, 0, -1):
        src = f"{log_path}.{idx}"
        dst = f"{log_path}.{idx + 1}"
        if os.path.exists(src):
            if idx == backups:
                os.remove(src)
            else:
                os.replace(src, dst)

    os.replace(log_path, f"{log_path}.1")


def _check_timeout():
    """Check if we've exceeded the maximum runtime."""
    elapsed = time.time() - _start_time
    if elapsed > MAX_RUNTIME_SECONDS:
        print(f"\n⚠ Runtime limit reached ({elapsed:.0f}s > {MAX_RUNTIME_SECONDS}s). Stopping.")
        return True
    return False


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
            action["diagnosis_codes"] = [str(c) for c in action["diagnosis_codes"][:5]]
            if "procedure_codes" not in action:
                action["procedure_codes"] = []
            if isinstance(action.get("procedure_codes"), str):
                action["procedure_codes"] = [action["procedure_codes"]]
            action["procedure_codes"] = [str(c) for c in action.get("procedure_codes", [])[:5]]
            if "decision" not in action:
                action["decision"] = "review"
            if action.get("decision", "").lower() not in ("approve", "reject", "review"):
                action["decision"] = "review"
            if "confidence" not in action:
                action["confidence"] = 0.5
            try:
                action["confidence"] = max(0.0, min(1.0, float(action["confidence"])))
            except (TypeError, ValueError):
                action["confidence"] = 0.5
            if "reasoning" not in action or len(str(action.get("reasoning", ""))) < 15:
                action["reasoning"] = "Medical coding assessment based on clinical documentation review and compliance guidelines."
            action["reasoning"] = str(action["reasoning"])[:500]
            if "modifier_codes" not in action:
                action["modifier_codes"] = []
            if isinstance(action.get("modifier_codes"), str):
                action["modifier_codes"] = [action["modifier_codes"]]
            action["modifier_codes"] = [str(c) for c in action.get("modifier_codes", [])[:3]]
            if "risk_flags" not in action:
                action["risk_flags"] = []
            if isinstance(action.get("risk_flags"), str):
                action["risk_flags"] = [action["risk_flags"]]
            action["risk_flags"] = [str(c) for c in action.get("risk_flags", [])[:5]]

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


# ──────────────────────────────────────────────
#  Structured logging helpers — [START] [STEP] [END]
# ──────────────────────────────────────────────

def log_start(task_id: str, metadata: Optional[dict] = None):
    """Emit a [START] structured log line."""
    entry = {"task_id": task_id}
    if metadata:
        entry.update(metadata)
    print(f"[START] {json.dumps(entry)}", flush=True)


def log_step(task_id: str, step: int, action: dict, reward: float, done: bool, info: Optional[dict] = None):
    """Emit a [STEP] structured log line."""
    reward = rounded_open_interval_score(reward, 4)
    entry = {
        "task_id": task_id,
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
    }
    if info:
        entry["info"] = info
    print(f"[STEP] {json.dumps(entry)}", flush=True)


def log_end(task_id: str, reward: float, metadata: Optional[dict] = None):
    """Emit an [END] structured log line."""
    reward = rounded_open_interval_score(reward, 4)
    metadata = dict(metadata or {})
    if not is_strict_open_interval(reward):
        reward = rounded_open_interval_score(0.0, 4)
        metadata["score_sanitized"] = True

    entry = {
        "task_id": task_id,
        "reward": reward,
    }
    if metadata:
        entry.update(metadata)
    print(f"[END] {json.dumps(entry)}", flush=True)


# ──────────────────────────────────────────────
#  Main evaluation loop
# ──────────────────────────────────────────────

def run_evaluation():
    """Run the baseline evaluation across all difficulty levels."""
    print("=" * 70)
    print("MedCodeRL - Baseline Inference Script")
    print("=" * 70)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Cases per difficulty: {CASES_PER_DIFFICULTY}")
    print(f"Max runtime: {MAX_RUNTIME_SECONDS}s")
    print("=" * 70)

    client = create_client()
    env = MyEnvironment()

    all_scores = []
    results_by_difficulty = {}

    for difficulty in ["easy", "medium", "hard"]:
        task_id = difficulty
        available = len(env._task_cases.get(difficulty, []))
        num_cases = min(CASES_PER_DIFFICULTY, available)
        task_ended = False

        if num_cases == 0:
            print(f"  No cases available for {difficulty}")
            fallback_task_score = rounded_open_interval_score(0.0, 4)
            results_by_difficulty[difficulty] = {
                "scores": [],
                "average": fallback_task_score,
                "count": 0,
            }

            # Emit structured task-level logs even when empty, so validators
            # never infer an implicit 0.0 score for a missing task.
            log_start(task_id, {"model": MODEL_NAME, "num_cases": 0})
            log_end(task_id, fallback_task_score, {"num_cases": 0, "skipped": "no_cases"})
            task_ended = True
            continue

        # Check timeout before starting a difficulty tier
        if _check_timeout():
            print(f"  Skipping {difficulty} due to runtime limit.")
            fallback_task_score = rounded_open_interval_score(0.0, 4)
            results_by_difficulty[difficulty] = {
                "scores": [],
                "average": fallback_task_score,
                "count": 0,
            }
            log_start(task_id, {"model": MODEL_NAME, "num_cases": 0})
            log_end(task_id, fallback_task_score, {"num_cases": 0, "skipped": "timeout"})
            task_ended = True
            continue

        # ── [START] ──
        log_start(task_id, {"model": MODEL_NAME, "num_cases": num_cases})

        print(f"\n{'─' * 50}")
        print(f"  Running {difficulty.upper()} tasks")
        print(f"{'─' * 50}")

        difficulty_scores = []

        try:
            for i in range(num_cases):
                # Check timeout before each case
                if _check_timeout():
                    break

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
                    score = to_open_interval_score(result_obs.reward if result_obs.reward is not None else 0.0)
                    done = result_obs.done if result_obs.done is not None else True
                    difficulty_scores.append(score)
                    all_scores.append(score)

                    # ── [STEP] ──
                    log_step(
                        task_id=task_id,
                        step=i + 1,
                        action=action_dict,
                        reward=rounded_open_interval_score(score, 4),
                        done=done,
                        info={
                            "case_id": obs.case_id,
                            "feedback": result_obs.feedback if result_obs.feedback else None,
                        },
                    )

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
                    fallback_score = to_open_interval_score(0.0)
                    difficulty_scores.append(fallback_score)
                    all_scores.append(fallback_score)

                    # ── [STEP] with failure ──
                    log_step(
                        task_id=task_id,
                        step=i + 1,
                        action=action_dict,
                        reward=fallback_score,
                        done=True,
                        info={"error": str(e)},
                    )

                # Rate limiting
                time.sleep(0.5)

            if difficulty_scores:
                avg = sum(difficulty_scores) / len(difficulty_scores)
                normalized_avg = rounded_open_interval_score(avg, 4)
                results_by_difficulty[difficulty] = {
                    "scores": difficulty_scores,
                    "average": normalized_avg,
                    "count": len(difficulty_scores),
                }
                print(f"\n  {difficulty.upper()} Average: {avg:.4f} ({len(difficulty_scores)} cases)")

                # ── [END] ──
                log_end(task_id, normalized_avg, {"num_cases": len(difficulty_scores)})
                task_ended = True
            else:
                # If a tier is interrupted before any scored step, still emit
                # a valid task score inside (0, 1) to satisfy strict validators.
                fallback_task_score = rounded_open_interval_score(0.0, 4)
                results_by_difficulty[difficulty] = {
                    "scores": [],
                    "average": fallback_task_score,
                    "count": 0,
                }
                print(f"\n  {difficulty.upper()} Average: {fallback_task_score:.4f} (0 cases)")
                log_end(task_id, fallback_task_score, {"num_cases": 0, "skipped": "no_scored_cases"})
                task_ended = True
        except Exception as difficulty_error:
            print(f"\n  ✗ Difficulty '{difficulty}' failed unexpectedly: {difficulty_error}")
            fallback_task_score = rounded_open_interval_score(0.0, 4)
            results_by_difficulty[difficulty] = {
                "scores": [],
                "average": fallback_task_score,
                "count": 0,
            }
            if not task_ended:
                log_end(
                    task_id,
                    fallback_task_score,
                    {"num_cases": 0, "skipped": "difficulty_exception", "error": str(difficulty_error)},
                )
                task_ended = True

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
        overall = rounded_open_interval_score(0.0, 4)
        print("\n  No scores recorded; using safe fallback overall score.")

    elapsed = time.time() - _start_time
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Write results to file (final normalization for strict open-interval compliance)
    normalized_results_by_difficulty = {}
    for diff, result in results_by_difficulty.items():
        normalized_scores = [rounded_open_interval_score(s, 4) for s in result.get("scores", [])]
        normalized_avg = rounded_open_interval_score(
            (sum(normalized_scores) / len(normalized_scores)) if normalized_scores else 0.0,
            4,
        )
        normalized_results_by_difficulty[diff] = {
            "scores": normalized_scores,
            "average": normalized_avg,
            "count": len(normalized_scores),
        }

    results_output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "cases_per_difficulty": CASES_PER_DIFFICULTY,
        "results_by_difficulty": normalized_results_by_difficulty,
        "overall_score": rounded_open_interval_score(overall, 4),
        "total_cases": len(all_scores),
        "runtime_seconds": round(elapsed, 1),
    }

    with open("baseline_results.json", "w") as f:
        json.dump(results_output, f, indent=2)
    print(f"\nResults saved to baseline_results.json")

    return overall


if __name__ == "__main__":
    log_path = _resolve_log_path()
    _rotate_log_if_needed(log_path)
    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    print(f"\n[LOG] Writing run output to {log_path}")
    try:
        score = run_evaluation()
        sys.exit(0 if score > 0 else 1)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
