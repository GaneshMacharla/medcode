---
title: MedCodeRL - Medical Coding & Billing Compliance Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7680
base_path: /web
tags:
  - openenv
---

# MedCodeRL 🏥

**Medical Coding & Billing Compliance OpenEnv Environment**

A realistic RL environment where AI agents must navigate the complex world of medical coding (ICD-10/CPT), billing compliance, and fraud detection. Built to the [OpenEnv specification](https://github.com/meta-pytorch/OpenEnv).

## 🎯 Why This Matters

- The US healthcare system loses **$125B+ annually** to incorrect medical coding
- Hospitals spend **$80K+ per coder** annually with 12–18 month training cycles
- Current LLMs fail at ICD-10/CPT coding because they lack **hierarchical constraint understanding**
- No existing OpenEnv environment covers this critical domain

## Quick Start

```python
from my_env import MedAction, MedCodeEnv

try:
    env = MedCodeEnv.from_docker_image("medcoderl:latest")

    result = env.reset()
    print(f"Case: {result.observation.case_id}")
    print(f"Note: {result.observation.clinical_note}")

    action = MedAction(
        diagnosis_codes=["J02.9"],
        procedure_codes=["99213"],
        decision="approve",
        confidence=0.9,
        reasoning="Acute pharyngitis with appropriate E&M coding for straightforward visit.",
        risk_flags=[]
    )
    result = env.step(action)
    print(f"Score: {result.reward}")

finally:
    env.close()
```

## 🐳 Building & Running with Docker

```bash
# Build from project root
docker build -t medcoderl .

# Run locally
docker run -p 7680:7680 medcoderl

# Verify it's running
curl http://localhost:7680/health
```

## Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face (Docker SDK):
```bash
# Via huggingface_hub CLI
huggingface-cli repo create medcoderl --type space --space-sdk docker
```

2. Push your code:
```bash
git remote add hf https://huggingface.co/spaces/<your-username>/medcoderl
git push hf main
```

3. Or use:
```bash
openenv push
```

## ⚙️ Environment Variables

The following variables **must** be set before running `inference.py`:

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | ✅ | Your Hugging Face / API key (also accepts `OPENAI_API_KEY`) |
| `API_BASE_URL` | ✅ | The API endpoint for the LLM (default: `https://api.openai.com/v1`) |
| `MODEL_NAME` | ✅ | The model identifier to use for inference (default: `gpt-4o-mini`) |
| `CASES_PER_DIFFICULTY` | ❌ | Number of cases per difficulty tier (default: `5`) |
| `MAX_RUNTIME_SECONDS` | ❌ | Timeout safety limit (default: `1100` — 18.3 min) |

## 🔬 Environment Details

### Action Space (MedAction)

| Field | Type | Description |
|---|---|---|
| `diagnosis_codes` | list[str] (1–5) | ICD-10-CM codes (primary + secondary) |
| `procedure_codes` | list[str] (0–5) | CPT/HCPCS procedure codes |
| `decision` | approve / reject / review | Billing compliance decision |
| `confidence` | float (0.0–1.0) | Agent confidence in coding decision |
| `reasoning` | str (15–500 chars) | Clinical justification |
| `modifier_codes` | list[str] (0–3) | Optional CPT modifier codes |
| `risk_flags` | list[str] (0–5) | Compliance risk flags identified |

### Observation Space (MedObservation)

| Field | Type | Description |
|---|---|---|
| `case_id` | str | Unique case identifier |
| `difficulty` | str | easy / medium / hard |
| `clinical_note` | str | Full clinical documentation |
| `symptoms` | list[str] | Reported symptoms |
| `treatments` | list[str] | Treatments administered |
| `insurance_type` | str | Medicare / Medicaid / Private / Uninsured |
| `prior_auth_required` | bool | Prior authorization needed |
| `treatment_cost` | str | low / medium / high |
| `patient_age` | int | Patient age |
| `patient_sex` | str | M / F |
| `provider_specialty` | str | Treating provider specialty |
| `visit_type` | str | inpatient / outpatient / emergency / telehealth |
| `comorbidities` | list[str] | Pre-existing conditions |
| `lab_results` | str / null | Relevant lab findings |
| `medications` | list[str] | Current medications |

### Reward System

**Grader Components (Deterministic, 0.0–1.0):**

| Component | Weight | Description |
|---|---|---|
| Diagnosis accuracy (ICD-10) | 35% | Jaccard + partial prefix matching |
| Procedure accuracy (CPT) | 20% | Jaccard + partial prefix matching |
| Decision accuracy | 25% | Exact match (1.0), partial credit for "review" (0.2–0.3) |
| Reasoning quality | 10% | Length + medical terminology density |
| Risk flag identification | 5% | Jaccard similarity with expected flags |
| Confidence calibration | 5% | |conf − correctness| penalty |

**Shaped Penalties** (scaled by difficulty — easy ×0.8, medium ×1.0, hard ×1.2):

| Penalty | Value | Trigger |
|---|---|---|
| Wrong approval | −0.25 | Approved a case that should be rejected |
| Wrong denial | −0.20 | Rejected a case that should be approved |
| Upcoding | −0.15 | Predicted >1 extra procedure codes |
| Missing primary code | −0.15 | Ground truth primary ICD-10 code not in prediction |
| Undercoding | −0.10 | <50% of expected diagnoses covered |
| Unnecessary procedure | −0.10 | Predicted procedures not in ground truth |
| Low confidence | −0.05 | Confidence < 0.2 |

**Bonuses:**

| Bonus | Value | Trigger |
|---|---|---|
| Perfect diagnosis | +0.05 | Diagnosis accuracy ≥ 0.99 |
| Good reasoning | +0.03 | Reasoning quality ≥ 0.80 |
| All risk flags | +0.05 | Risk identification ≥ 0.99 |

## 📋 Tasks (90 cases total)

### 🟢 Easy (30 cases)
Straightforward clinical cases with single diagnoses and direct ICD-10/CPT mapping.
Examples: viral pharyngitis, UTI, ankle sprain, routine wellness exam, vaccination, tension headache.

### 🟡 Medium (30 cases)
Multi-diagnosis cases with comorbidities, insurance considerations, and partial ambiguity.
Examples: COPD with pneumonia, diabetic neuropathy, cardiac workup, RA flare, MS relapse, hip fracture.

### 🔴 Hard (30 cases)
Complex compliance dilemmas: upcoding, unbundling, fraud detection, medically unnecessary treatments, dangerous polypharmacy, ethical edge cases.
Examples: Medicare fraud (cloned notes, unbundled labs), off-label immunotherapy, DKA in uninsured patient, advanced dementia with aggressive intervention requests.

## 🚀 Running the Inference Script

```bash
export HF_TOKEN="your-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

### Structured Logging

The inference script emits **structured stdout logs** in `[START]`, `[STEP]`, `[END]` format as required by the OpenEnv evaluation pipeline:

```
[START] {"task_id": "easy", "model": "gpt-4o-mini", "num_cases": 5}
[STEP]  {"task_id": "easy", "step": 1, "action": {...}, "reward": 0.72, "done": true, "info": {"case_id": "easy_001", "feedback": "..."}}
[STEP]  {"task_id": "easy", "step": 2, "action": {...}, "reward": 0.65, "done": true, "info": {"case_id": "easy_002", "feedback": "..."}}
...
[END]   {"task_id": "easy", "reward": 0.68, "num_cases": 5}
[START] {"task_id": "medium", "model": "gpt-4o-mini", "num_cases": 5}
...
[END]   {"task_id": "medium", "reward": 0.52, "num_cases": 5}
[START] {"task_id": "hard", "model": "gpt-4o-mini", "num_cases": 5}
...
[END]   {"task_id": "hard", "reward": 0.31, "num_cases": 5}
```

### Expected Baseline Scores (gpt-4o-mini)

| Difficulty | Expected Avg | Score Range |
|---|---|---|
| Easy | ~0.70 | 0.55 – 0.85 |
| Medium | ~0.50 | 0.35 – 0.65 |
| Hard | ~0.30 | 0.15 – 0.45 |
| **Overall** | **~0.50** | **0.35 – 0.65** |

## Development & Testing

### Run comprehensive tests

```bash
python test_env.py
```

Runs 11 tests covering case loading, reset/step/state API, reward range, grader determinism, reward shaping, episode boundaries, and invalid action handling.

### Run server locally

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 7680
```

### Direct environment testing

```python
from server.my_env_environment import MyEnvironment
from models import MedAction

env = MyEnvironment()
obs = env.reset(task_id="easy")
action = MedAction(
    diagnosis_codes=["J02.9"],
    procedure_codes=["99213"],
    decision="approve",
    confidence=0.9,
    reasoning="Acute pharyngitis with appropriate coding.",
    risk_flags=[]
)
result = env.step(action)
print(f"Score: {result.reward}, Done: {result.done}")
```

### Pre-submission validation

```bash
# Validate locally
./validate-submission.sh https://your-space.hf.space

# Or run openenv validate directly
openenv validate
```

## Project Structure

```
medcoderl/
├── __init__.py              # Module exports
├── README.md                # This file
├── openenv.yaml             # OpenEnv manifest (full metadata)
├── pyproject.toml           # Dependencies
├── Dockerfile               # Root Dockerfile for HF Spaces
├── client.py                # MedCodeEnv client
├── models.py                # MedAction & MedObservation models
├── inference.py             # Baseline inference script (structured logging)
├── test_env.py              # Comprehensive environment tests (11 tests)
├── validate-submission.sh   # Pre-submission validator
├── tasks/
│   ├── easy.json            # 30 easy clinical cases
│   ├── medium.json          # 30 medium clinical cases
│   └── hard.json            # 30 hard clinical cases
└── server/
    ├── __init__.py           # Server exports
    ├── my_env_environment.py # Core env logic + grader + rewards
    ├── app.py                # FastAPI application
    ├── Dockerfile            # Alternative multi-stage Dockerfile
    └── requirements.txt      # Server dependencies
```

## ⚠️ Disclaimer

This environment is a **simulation for AI training and evaluation only**. It does not use real patient data and should not be used for actual medical coding or billing. All clinical cases are synthetic.

## License

MIT License
