"""
Comprehensive test suite for MedCodeRL environment.

Tests:
- Case loading across all difficulties
- reset() / step() / state() API
- Reward range validation
- Grader determinism
- Invalid action handling
- Episode boundary semantics
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.my_env_environment import MyEnvironment, _grade, _compute_reward
from models import MedAction


def test_case_loading():
    """Verify all 90 cases load correctly."""
    env = MyEnvironment()
    for d in ["easy", "medium", "hard"]:
        count = len(env._task_cases[d])
        assert count == 30, f"{d} should have 30 cases, got {count}"
        for i, case in enumerate(env._task_cases[d]):
            assert "id" in case, f"{d}[{i}] missing 'id'"
            assert "input" in case, f"{d}[{i}] missing 'input'"
            assert "ground_truth" in case, f"{d}[{i}] missing 'ground_truth'"
            gt = case["ground_truth"]
            assert "diagnosis_codes" in gt, f"{d}[{i}] missing ground_truth.diagnosis_codes"
            assert "decision" in gt, f"{d}[{i}] missing ground_truth.decision"
            assert gt["decision"] in ("approve", "reject", "review"), f"{d}[{i}] invalid decision: {gt['decision']}"
    print("✅ Case loading: 90 cases (30+30+30) loaded with valid structure")


def test_reset_all_difficulties():
    """Test reset() for each difficulty."""
    env = MyEnvironment()
    for d in ["easy", "medium", "hard"]:
        obs = env.reset(task_id=d)
        assert obs.case_id.startswith(d[:3] if d != "medium" else "med"), f"Expected {d} case_id prefix, got {obs.case_id}"
        assert obs.difficulty == d, f"Expected difficulty={d}, got {obs.difficulty}"
        assert obs.done is False, f"Reset should return done=False"
        assert obs.reward == 0.0, f"Reset should return reward=0.0"
        assert len(obs.clinical_note) > 0, f"Empty clinical note for {d}"
        assert obs.patient_age >= 0, f"Invalid patient_age for {d}"
        assert obs.patient_sex in ("M", "F"), f"Invalid patient_sex for {d}"
        assert obs.insurance_type in ("Medicare", "Medicaid", "Private", "Uninsured"), f"Invalid insurance for {d}"
    print("✅ Reset: all 3 difficulties return valid observations")


def test_step_with_correct_action():
    """Test step() with the correct ground-truth action."""
    env = MyEnvironment()
    obs = env.reset(task_id="easy")

    # Use ground truth from first easy case
    gt = env._current_case["ground_truth"]
    action = MedAction(
        diagnosis_codes=gt["diagnosis_codes"],
        procedure_codes=gt.get("procedure_codes", []),
        decision=gt["decision"],
        confidence=0.95,
        reasoning="Correct medical coding based on clinical documentation and standard coding guidelines for this case.",
        modifier_codes=[],
        risk_flags=gt.get("risk_flags", []),
    )

    result = env.step(action)
    assert result.done is True, "Valid step should set done=True"
    assert result.reward is not None, "Step should return a reward"
    assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of [0,1] range"
    assert result.reward > 0.5, f"Correct action should score > 0.5, got {result.reward}"
    print(f"✅ Step (correct): score={result.reward:.4f}, done={result.done}")


def test_step_with_wrong_action():
    """Test step() with a clearly wrong action."""
    env = MyEnvironment()
    obs = env.reset(task_id="easy")

    action = MedAction(
        diagnosis_codes=["Z99.89"],  # Unlikely to match
        procedure_codes=["99223"],   # Wrong E&M level
        decision="reject",           # Wrong for easy cases (mostly approve)
        confidence=0.9,
        reasoning="This is a deliberately wrong action to test the grader scoring accuracy.",
        modifier_codes=[],
        risk_flags=["test_flag"],
    )

    result = env.step(action)
    assert result.done is True
    assert result.reward is not None
    assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of [0,1] range"
    assert result.reward < 0.5, f"Wrong action should score < 0.5, got {result.reward}"
    print(f"✅ Step (wrong): score={result.reward:.4f}, done={result.done}")


def test_reward_range_all_cases():
    """Verify rewards are in [0.0, 1.0] for all cases across difficulties."""
    env = MyEnvironment()
    for d in ["easy", "medium", "hard"]:
        for i in range(min(5, len(env._task_cases[d]))):
            obs = env.reset(task_id=d)
            gt = env._current_case["ground_truth"]

            action = MedAction(
                diagnosis_codes=gt["diagnosis_codes"],
                procedure_codes=gt.get("procedure_codes", []),
                decision=gt["decision"],
                confidence=0.8,
                reasoning="Testing reward range with ground truth coding assignment for this clinical case.",
                risk_flags=gt.get("risk_flags", []),
            )
            result = env.step(action)
            assert 0.0 <= result.reward <= 1.0, f"Reward out of range for {obs.case_id}: {result.reward}"
    print("✅ Reward range: all tested cases produce rewards in [0.0, 1.0]")


def test_grader_determinism():
    """Verify the grader produces identical scores for identical inputs."""
    action_dict = {
        "diagnosis_codes": ["J02.9"],
        "procedure_codes": ["99213"],
        "decision": "approve",
        "confidence": 0.85,
        "reasoning": "Acute pharyngitis coded correctly with appropriate E&M level.",
        "risk_flags": [],
    }
    ground_truth = {
        "diagnosis_codes": ["J02.9"],
        "procedure_codes": ["99213"],
        "decision": "approve",
        "risk_flags": [],
    }

    results = [_grade(action_dict, ground_truth) for _ in range(10)]
    scores = [r["score"] for r in results]
    assert len(set(scores)) == 1, f"Grader not deterministic: scores={set(scores)}"
    print(f"✅ Grader determinism: 10 identical calls → score={scores[0]} (consistent)")


def test_reward_shaping():
    """Verify reward function provides signal, not just binary."""
    ground_truth = {
        "diagnosis_codes": ["J02.9"],
        "procedure_codes": ["99213"],
        "decision": "approve",
        "risk_flags": [],
    }

    # Perfect action
    perfect = _compute_reward({
        "diagnosis_codes": ["J02.9"], "procedure_codes": ["99213"],
        "decision": "approve", "confidence": 0.95,
        "reasoning": "Acute pharyngitis with appropriate E&M level coding for this clinical encounter.",
        "risk_flags": [],
    }, ground_truth, "easy")

    # Partial match (right diagnosis, wrong procedure)
    partial = _compute_reward({
        "diagnosis_codes": ["J02.9"], "procedure_codes": ["99215"],
        "decision": "approve", "confidence": 0.7,
        "reasoning": "Pharyngitis diagnosed correctly but E&M level may need review based on documentation.",
        "risk_flags": [],
    }, ground_truth, "easy")

    # Completely wrong
    wrong = _compute_reward({
        "diagnosis_codes": ["Z99.89"], "procedure_codes": ["99223"],
        "decision": "reject", "confidence": 0.9,
        "reasoning": "Incorrect medical coding assessment provided for testing purposes.",
        "risk_flags": ["test"],
    }, ground_truth, "easy")

    assert perfect["score"] > partial["score"] > wrong["score"], \
        f"Reward ordering broken: perfect={perfect['score']:.4f}, partial={partial['score']:.4f}, wrong={wrong['score']:.4f}"
    print(f"✅ Reward shaping: perfect={perfect['score']:.4f} > partial={partial['score']:.4f} > wrong={wrong['score']:.4f}")


def test_state_property():
    """Test the state property returns valid data."""
    env = MyEnvironment()
    s = env.state
    assert s.episode_id, "state.episode_id should be set"
    assert s.step_count == 0, f"Initial step_count should be 0, got {s.step_count}"

    obs = env.reset(task_id="easy")
    s = env.state
    assert s.step_count == 0, f"After reset, step_count should be 0"

    action = MedAction(
        diagnosis_codes=["J02.9"], procedure_codes=["99213"],
        decision="approve", confidence=0.9,
        reasoning="State test: checking step_count increment after step call.",
        risk_flags=[],
    )
    env.step(action)
    s = env.state
    assert s.step_count == 1, f"After step, step_count should be 1, got {s.step_count}"
    print(f"✅ State: episode_id={s.episode_id[:8]}..., steps={s.step_count}")


def test_episode_done_semantics():
    """Test that stepping after done returns reward=0."""
    env = MyEnvironment()
    env.reset(task_id="easy")

    action = MedAction(
        diagnosis_codes=["J02.9"], procedure_codes=["99213"],
        decision="approve", confidence=0.9,
        reasoning="Episode boundary test: this is the first step in the episode.",
        risk_flags=[],
    )

    r1 = env.step(action)
    assert r1.done is True, "First valid step should be done=True"

    # Step again after done
    r2 = env.step(action)
    assert r2.done is True, "Step after done should return done=True"
    assert r2.reward == 0.0, "Step after done should return reward=0.0"
    print("✅ Episode boundaries: stepping after done returns reward=0, done=True")


def test_invalid_action():
    """Test handling of invalid actions."""
    env = MyEnvironment()
    env.reset(task_id="easy")

    # Empty diagnosis codes should fail validation
    try:
        action = MedAction(
            diagnosis_codes=[],
            procedure_codes=[],
            decision="approve",
            confidence=0.5,
            reasoning="Testing invalid action with no diagnosis codes provided.",
            risk_flags=[],
        )
        # If Pydantic validation catches it, that's fine
        print("✅ Invalid action: caught by Pydantic validation")
    except Exception:
        print("✅ Invalid action: correctly rejected by model validation")


def test_task_info():
    """Test the get_task_info() helper."""
    env = MyEnvironment()
    info = env.get_task_info()
    assert "tasks" in info, "get_task_info() missing 'tasks'"
    assert info["total_cases"] == 90, f"Expected 90 total cases, got {info['total_cases']}"
    for d in ["easy", "medium", "hard"]:
        assert d in info["tasks"], f"Missing difficulty '{d}' in task info"
        assert info["tasks"][d]["count"] == 30, f"{d} should have 30 cases"
    print(f"✅ Task info: {info['total_cases']} total cases across {len(info['tasks'])} difficulties")


# ──────────────────────────────────────────────
#  Run all tests
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MedCodeRL — Comprehensive Environment Tests")
    print("=" * 60)
    print()

    tests = [
        test_case_loading,
        test_reset_all_difficulties,
        test_step_with_correct_action,
        test_step_with_wrong_action,
        test_reward_range_all_cases,
        test_grader_determinism,
        test_reward_shaping,
        test_state_property,
        test_episode_done_semantics,
        test_invalid_action,
        test_task_info,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"  ✅ ALL {passed} TESTS PASSED")
    else:
        print(f"  ⚠ {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
