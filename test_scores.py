"""Test script to check if any reward values fall outside strict (0, 1) interval."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.my_env_environment import MyEnvironment, _grade, _compute_reward, _to_open_interval_score
from models import MedAction

env = MyEnvironment()

violations = []

# Test 1: Check all reset rewards
print("=" * 60)
print("TEST 1: Checking reset rewards for all tasks")
print("=" * 60)
for diff in ["easy", "medium", "hard"]:
    cases = env._task_cases.get(diff, [])
    for i, case in enumerate(cases):
        obs = env.reset(task_id=case["id"])
        r = obs.reward
        if r is None or r <= 0.0 or r >= 1.0:
            violations.append(f"RESET {case['id']}: reward={r}")
            print(f"  ❌ {case['id']}: reward={r}")
        else:
            print(f"  ✓ {case['id']}: reward={r}")

# Test 2: Check step rewards with fallback action
print("\n" + "=" * 60)
print("TEST 2: Checking step rewards with fallback action")
print("=" * 60)
fallback = MedAction(
    diagnosis_codes=["R69"],
    procedure_codes=["99213"],
    decision="review",
    confidence=0.1,
    reasoning="Unable to obtain LLM response. Flagging for manual review as a safety measure.",
    modifier_codes=[],
    risk_flags=["llm_failure"],
)

for diff in ["easy", "medium", "hard"]:
    cases = env._task_cases.get(diff, [])
    for i, case in enumerate(cases):
        env.reset(task_id=case["id"])
        obs = env.step(fallback)
        r = obs.reward
        if r is None or r <= 0.0 or r >= 1.0:
            violations.append(f"STEP-FALLBACK {case['id']}: reward={r}")
            print(f"  ❌ {case['id']}: reward={r}")
        else:
            print(f"  ✓ {case['id']}: reward={r}")

# Test 3: Check step rewards with perfect action (using ground truth)
print("\n" + "=" * 60)
print("TEST 3: Checking step rewards with ground truth (perfect) action")
print("=" * 60)
for diff in ["easy", "medium", "hard"]:
    cases = env._task_cases.get(diff, [])
    for i, case in enumerate(cases):
        gt = case.get("ground_truth", {})
        env.reset(task_id=case["id"])
        perfect = MedAction(
            diagnosis_codes=gt.get("diagnosis_codes", ["R69"]),
            procedure_codes=gt.get("procedure_codes", ["99213"]),
            decision=gt.get("decision", "review"),
            confidence=0.9,
            reasoning="Based on clinical documentation review, ICD-10 coding guidelines, and CPT procedure validation. The diagnosis codes align with the documented symptoms and clinical findings. Billing compliance verified.",
            modifier_codes=[],
            risk_flags=gt.get("risk_flags", []),
        )
        obs = env.step(perfect)
        r = obs.reward
        if r is None or r <= 0.0 or r >= 1.0:
            violations.append(f"STEP-PERFECT {case['id']}: reward={r}")
            print(f"  ❌ {case['id']}: reward={r}")
        else:
            print(f"  ✓ {case['id']}: reward={r}")

# Test 4: Check step rewards with invalid action
print("\n" + "=" * 60)
print("TEST 4: Checking step rewards with invalid action")
print("=" * 60)
for diff in ["easy", "medium", "hard"]:
    cases = env._task_cases.get(diff, [])
    for case in cases[:3]:  # Just check a few
        env.reset(task_id=case["id"])
        invalid = MedAction(
            diagnosis_codes=["INVALID"],
            procedure_codes=[],
            decision="review",
            confidence=0.5,
            reasoning="This is a test reasoning that is long enough to pass validation.",
            modifier_codes=[],
            risk_flags=[],
        )
        obs = env.step(invalid)
        r = obs.reward
        if r is None or r <= 0.0 or r >= 1.0:
            violations.append(f"STEP-INVALID {case['id']}: reward={r}")
            print(f"  ❌ {case['id']}: reward={r}")
        else:
            print(f"  ✓ {case['id']}: reward={r}")

# Test 5: Check raw grading for edge cases
print("\n" + "=" * 60)
print("TEST 5: Checking raw _grade and _compute_reward edge cases")
print("=" * 60)

# Worst case: everything wrong
worst = {
    "diagnosis_codes": ["Z99"],
    "procedure_codes": ["00000"],
    "decision": "approve",
    "confidence": 0.0,
    "reasoning": "short reason text for testing the grading system output",
    "modifier_codes": [],
    "risk_flags": [],
}
worst_gt = {
    "diagnosis_codes": ["A01.0", "B02.1"],
    "procedure_codes": ["99214"],
    "decision": "reject",
    "confidence": 0.8,
    "risk_flags": ["upcoding_risk", "missing_documentation"],
}
grade = _grade(worst, worst_gt)
print(f"  Worst grade: {grade}")
for k, v in grade.items():
    if v <= 0.0 or v >= 1.0:
        violations.append(f"GRADE worst.{k}: {v}")
        print(f"  ❌ {k}={v}")

reward = _compute_reward(worst, worst_gt, "hard")
print(f"  Worst reward: {reward['score']}")
if reward['score'] <= 0.0 or reward['score'] >= 1.0:
    violations.append(f"REWARD worst: {reward['score']}")
    print(f"  ❌ reward={reward['score']}")

# Best case: everything matches
best_gt2 = {
    "diagnosis_codes": ["J06.9"],
    "procedure_codes": ["99213"],
    "decision": "approve",
    "risk_flags": ["none"],
}
best = {
    "diagnosis_codes": ["J06.9"],
    "procedure_codes": ["99213"],
    "decision": "approve",
    "confidence": 0.9,
    "reasoning": "Based on clinical documentation, ICD-10 coding guidelines, CPT procedure codes are medically necessary and comply with billing standards.",
    "modifier_codes": [],
    "risk_flags": ["none"],
}
grade2 = _grade(best, best_gt2)
print(f"  Best grade: {grade2}")
for k, v in grade2.items():
    if v <= 0.0 or v >= 1.0:
        violations.append(f"GRADE best.{k}: {v}")
        print(f"  ❌ {k}={v}")

reward2 = _compute_reward(best, best_gt2, "easy")
print(f"  Best reward: {reward2['score']}")
if reward2['score'] <= 0.0 or reward2['score'] >= 1.0:
    violations.append(f"REWARD best: {reward2['score']}")
    print(f"  ❌ reward={reward2['score']}")

# Test 6: Direct _to_open_interval_score edge cases
print("\n" + "=" * 60)
print("TEST 6: _to_open_interval_score edge cases")
print("=" * 60)
edge_values = [0.0, 1.0, -1.0, 2.0, 0.5, None, float('inf'), float('-inf'), float('nan')]
for v in edge_values:
    try:
        result = _to_open_interval_score(v)
        if result <= 0.0 or result >= 1.0:
            violations.append(f"_to_open_interval_score({v}): {result}")
            print(f"  ❌ _to_open_interval_score({v}) = {result}")
        else:
            print(f"  ✓ _to_open_interval_score({v}) = {result}")
    except Exception as e:
        print(f"  ❌ _to_open_interval_score({v}) EXCEPTION: {e}")
        violations.append(f"_to_open_interval_score({v}): EXCEPTION {e}")

# Summary
print("\n" + "=" * 60)
if violations:
    print(f"❌ FOUND {len(violations)} VIOLATIONS:")
    for v in violations:
        print(f"  - {v}")
else:
    print("✅ ALL SCORES ARE STRICTLY IN (0, 1)")
print("=" * 60)