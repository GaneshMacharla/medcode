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
    env = MedCodeEnv.from_docker_image("my_env-env:latest")

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

## Building the Docker Image

```bash
docker build -t my_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## 🔬 Environment Details

### Action (MedAction)

| Field | Type | Description |
|---|---|---|
| `diagnosis_codes` | list[str] (1–5) | ICD-10-CM codes |
| `procedure_codes` | list[str] (0–5) | CPT/HCPCS codes |
| `decision` | approve/reject/review | Billing compliance decision |
| `confidence` | float (0.0–1.0) | Agent confidence |
| `reasoning` | str (15–500 chars) | Clinical justification |
| `modifier_codes` | list[str] (0–3) | Optional CPT modifiers |
| `risk_flags` | list[str] (0–5) | Compliance risk flags |

### Observation (MedObservation)

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
| `provider_specialty` | str | Provider specialty |
| `visit_type` | str | inpatient / outpatient / emergency / telehealth |
| `comorbidities` | list[str] | Pre-existing conditions |
| `lab_results` | str/None | Lab findings |
| `medications` | list[str] | Current medications |

### Reward System

**Grader Components (Deterministic, 0.0–1.0):**

| Component | Weight |
|---|---|
| Diagnosis accuracy (ICD-10) | 35% |
| Procedure accuracy (CPT) | 20% |
| Decision accuracy | 25% |
| Reasoning quality | 10% |
| Risk flag identification | 5% |
| Confidence calibration | 5% |

**Shaped Penalties** (scaled by difficulty):
- Wrong approval: -0.25 | Wrong denial: -0.20
- Upcoding: -0.15 | Missing primary code: -0.15
- Undercoding: -0.10 | Unnecessary procedure: -0.10

**Bonuses:** Perfect diagnosis +0.05, Good reasoning +0.03, All flags +0.05

## 📋 Tasks (90 cases total)

### 🟢 Easy (30 cases)
Straightforward clinical cases with single diagnoses and direct ICD-10/CPT mapping.
Examples: viral pharyngitis, UTI, ankle sprain, routine wellness exam.

### 🟡 Medium (30 cases)
Multi-diagnosis cases with comorbidities, insurance considerations, and partial ambiguity.
Examples: COPD with pneumonia, diabetic neuropathy, cardiac workup.

### 🔴 Hard (30 cases)
Complex compliance dilemmas: upcoding, unbundling, fraud detection, medically unnecessary treatments, dangerous polypharmacy, ethical edge cases.

## 🚀 Running the Inference Script

```bash
export HF_TOKEN="your-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

### Expected Baseline Scores

| Difficulty | Score Range |
|---|---|
| Easy | 0.55 – 0.85 |
| Medium | 0.35 – 0.65 |
| Hard | 0.15 – 0.45 |

## Development & Testing

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

## Project Structure

```
my_env/
├── __init__.py              # Module exports
├── README.md                # This file
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── client.py                # MedCodeEnv client
├── models.py                # MedAction & MedObservation models
├── inference.py             # Baseline inference script
├── tasks/
│   ├── easy.json            # 30 easy clinical cases
│   ├── medium.json          # 30 medium clinical cases
│   └── hard.json            # 30 hard clinical cases
└── server/
    ├── __init__.py           # Server exports
    ├── my_env_environment.py # Core env logic + grader + rewards
    ├── app.py                # FastAPI application
    ├── Dockerfile            # Container image
    └── requirements.txt      # Server dependencies
```

## ⚠️ Disclaimer

This environment is a **simulation for AI training and evaluation only**. It does not use real patient data and should not be used for actual medical coding or billing. All clinical cases are synthetic.

## License

MIT License
