"""Quick test script for MedCodeRL in my_env template."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.my_env_environment import MyEnvironment
from models import MedAction

env = MyEnvironment()

# Check tasks loaded
print("Tasks loaded:")
for d in ["easy", "medium", "hard"]:
    print(f"  {d}: {len(env._task_cases[d])} cases")

# Test reset (easy)
obs = env.reset(task_id="easy")
print(f"\nReset OK - case: {obs.case_id}")
print(f"Clinical note: {obs.clinical_note[:80]}...")

# Test step with correct action
action = MedAction(
    diagnosis_codes=["J02.9"],
    procedure_codes=["99213"],
    decision="approve",
    confidence=0.85,
    reasoning="Acute pharyngitis coded correctly with appropriate E&M level for straightforward visit.",
    risk_flags=[],
)
result = env.step(action)
print(f"\nStep OK - Score: {result.reward}, Done: {result.done}")
print(f"Feedback: {result.feedback}")
print(f"State: episode={env.state.episode_id[:8]}... steps={env.state.step_count}")

# Test hard case with wrong action
obs2 = env.reset(task_id="hard")
print(f"\nHard case: {obs2.case_id}")
bad = MedAction(
    diagnosis_codes=["M17.11"],
    procedure_codes=["27447"],
    decision="approve",
    confidence=0.9,
    reasoning="Patient needs knee replacement as documented by the provider notes.",
    risk_flags=[],
)
r2 = env.step(bad)
print(f"Wrong action score: {r2.reward}")

# Test medium
obs3 = env.reset(task_id="medium")
print(f"\nMedium case: {obs3.case_id}")

print("\n✅ All tests passed!")
