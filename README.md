# AI Incident Judgment & Scoring Service

An automated, enterprise-grade FastAPI service that evaluates technical incident reports for **Report Quality** (0–100) and assesses **Incident Severity** (SEV1–SEV4) with mathematically calibrated confidence bounding, deterministic policy floors, and automated property-based evaluation.

---

## 1. Executive Summary

### Core Architectural Thesis
Large Language Models (LLMs) are powerful semantic perceptual engines but are fundamentally unreliable calculators. When tasked with direct numerical evaluation, LLMs exhibit severe failure modes: floating-point arithmetic drift, non-reproducible scoring, subjective severity inflation, and uncalibrated overconfidence.

**Our Core Thesis:**
> **The LLM is used strictly for semantic perception and feature extraction; pure, deterministic Python handles all arithmetic, weighting, normalization, policy floors, and confidence bounding.**

- **LLM Responsibility (Semantic Perception)**: Reads raw, unstructured text (narratives, stack traces, metrics, logs). Categorizes qualitative signals into discrete ordinals (0–5 dimension scores, ordinal impact/criticality levels), identifies specific information gaps, extracts direct evidence quotes, and formulates qualitative justifications **before** numerical output to preserve autoregressive chain-of-thought conditioning.
- **Python Responsibility (Deterministic Enforcement)**: Implements mathematical scoring equations, linear normalization, strict quality bands, non-negotiable policy floors (e.g., guaranteed SEV1 for credential leaks), evidence-calibrated confidence penalties, and deterministic triggers for human-in-the-loop review.

### Zero-Framework Domain Decoupling
The core domain package [`app/judgment/`](app/judgment/) maintains **ZERO framework (FastAPI, Starlette) and ZERO HTTP (httpx, requests, aiohttp) dependencies**. 

```
┌────────────────────────────────────────────────────────┐
│               FastAPI REST Layer (app/api/)            │
│       HTTP routing, Pydantic schemas, dependency injection
└───────────────────────────┬────────────────────────────┘
                            │ Calls domain orchestrator
                            ▼
┌────────────────────────────────────────────────────────┐
│          Domain Orchestrator (app/judgment/service.py) │
│       Pure Python orchestrator with zero HTTP/web deps  │
├───────────────────────────┬────────────────────────────┤
│   Semantic Perception     │    Deterministic Python    │
│   (app/judgment/llm/)     │  (scoring.py & conf.py)    │
│                           │                            │
│  - Structured extraction  │  - Quality weighted sum    │
│  - Ordinal signals        │  - Policy floor elevation  │
│  - Information gap codes  │  - Evidence ceiling bounds │
│  - Rationale conditioning │  - Clamped [0.05, 0.95]    │
└───────────────────────────┴────────────────────────────┘
```

Because `app/judgment/` is completely decoupled from the web layer:
1. It can be invoked directly in asynchronous worker queues (Celery, RQ, ARQ).
2. It can execute inside serverless functions (AWS Lambda, Google Cloud Run) or CLI pipelines.
3. Unit and property tests run in sub-millisecond offline execution without mocking HTTP connections or web frameworks.

### Architectural Pipeline Diagram

```
+-----------------------------------------------------------------------------------+
|                            Incoming Incident Report                               |
|          { title, description, impact?, evidence?, actions_taken? }               |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                         REST Layer (app/api/v1/router.py)                         |
|   - Pydantic schema validation (IncidentInput)                                    |
|   - Dependency injection (AssessmentService)                                      |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                    Domain Orchestrator (app/judgment/service.py)                  |
|   - ZERO framework or HTTP dependencies                                           |
|   - Input sanitization & XML-delimited prompt escaping (prompts.py)               |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     LLM Semantic Perception (app/judgment/llm/)                   |
|   - Providers: FakeLLMClient (offline/tests) | OpenAILLMClient (gpt-4o-mini)      |
|   - Strict output contract (LLMAssessment): justifications before ordinal scores  |
|   - Extracts: 5 Quality Dimensions (0-5), Severity Signals, Missing Info Codes    |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                   Deterministic Python Judgment & Calibration                     |
|                                                                                   |
|  +---------------------------+  +---------------------------+  +---------------+  |
|  | Quality Scoring Engine    |  | Severity & Policy Floors  |  | Mathematical  |  |
|  | (scoring.py)              |  | (scoring.py)              |  | Calibration   |  |
|  |                           |  |                           |  | (conf.py)     |  |
|  | - Weighted sum (0-100)    |  | - Index: 0.45*I+0.35*C... |  | - Ev. Ceiling |  |
|  | - Bands: POOR/WEAK/       |  | - SEV1 Data Breach Floor  |  | - Gap Penalty |  |
|  |   ADEQUATE/STRONG         |  | - SEV2 Total Outage Floor |  | - Clamped     |  |
|  +---------------------------+  +---------------------------+  +---------------+  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                      Human Review Dispatch & Final Result                         |
|   - Deterministic triggers: LOW confidence | POOR quality | Policy floor | SEV1   |
|   - Returns structured domain AssessmentResult -> Serialized AssessmentOut        |
+-----------------------------------------------------------------------------------+
```

---

## 2. Zero-Key Quickstart

### Prerequisites
- Python 3.11+
- (Optional) Docker

### 1. Local Setup
Clone the repository, create a virtual environment, and install dependencies:

```bash
# Clone and enter directory
git clone https://github.com/ehsan-hn/ReportJudge.git
cd ReportJudge

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Immediate Offline Execution (Zero API Keys Needed)
The service ships with a fully deterministic `FakeLLMClient` that extracts semantic signals, enforces contract invariants, and triggers policy floors without making any external API calls:

```bash
# Set provider to fake and launch
export LLM_PROVIDER=fake
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Interactive Documentation (Swagger UI)
Open your browser and navigate to:
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI JSON Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Health Endpoint**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
*(Note: Requesting the root URL `http://localhost:8000/` automatically redirects to `/docs`).*

#### Example Assessment Request (cURL)
```bash
curl -X POST http://localhost:8000/api/v1/assessments \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Auth service 500 error cascade",
    "description": "Auth cluster experiencing 100% failure rate on login endpoints across all regions.",
    "impact": "All users globally locked out; zero authentications succeeding.",
    "evidence": "Datadog alert: auth-service 5xx rate at 100% over 15m; zero tokens issued.",
    "actions_taken": "Restarted pods and rolled back release v2.4.1."
  }'
```

### 4. Switching to OpenAI
To switch from offline simulation to live OpenAI inference (`gpt-4o-mini` by default):

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"
uvicorn app.main:app --reload
```

### 5. Docker Build and Run
The application is fully containerized using a hardened `python:3.11-slim` base image with non-root security:

```bash
# Build the Docker image
docker build -t incident-judge .

# Run container in offline mode (default)
docker run -d -p 8000:8000 --name incident-judge-app incident-judge

# Run container with live OpenAI provider
docker run -d -p 8000:8000 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY="sk-your-openai-api-key" \
  --name incident-judge-app incident-judge

# Inspect container health
docker inspect --format='{{json .State.Health}}' incident-judge-app
```

---

## 3. Orthogonal Axes (Report Quality vs Incident Severity)

A foundational flaw in conventional incident triage systems is conflating **how bad the incident is** with **how well the incident was written up**. This service models them as two strictly independent, orthogonal axes.

```
       ▲ High
       │
       │    Scenario A: "Frantic Slack"          Scenario C: "Post-Mortem Grade"
       │    - Catastrophic SEV1 Data Breach      - Catastrophic SEV1 Data Breach
       │    - Vague 1-sentence writeup           - Rigorous telemetry & logs
       │    - Quality: POOR (24.0)               - Quality: STRONG (92.0)
       │    - Human Review: YES                  - Human Review: YES (SEV1 Policy)
INCIDENT
SEVERITY
       │    Scenario D: "Vague Complaint"        Scenario B: "Pristine CSS Bug"
       │    - Minor performance gripe            - Cosmetic 8px CSS overlap
       │    - No logs, no metrics                - 5 pages of HAR, screenshots, steps
       │    - Quality: POOR (18.0)               - Quality: STRONG (88.0)
       │    - Human Review: YES (Low Conf)       - Human Review: NO
       │
       └────────────────────────────────────────────────────────────────────────►
       Low                               Quality Band                       High
                                        REPORT QUALITY
```

### Scenario Comparison

#### Scenario A: Vague Catastrophic SEV1 (High Severity, Low Quality)
- **Input**: `"prod db credentials leaked on public pastebin, someone help asap!"`
- **Severity**: **`SEV1`** (Enforced by `FLOOR_DATA_LOSS_OR_BREACH_SEV1`).
- **Report Quality**: **`POOR` (20.0 / 100)**. Lacks impacted services, access logs, remediation steps, or timeline.
- **Triage Action**: Immediately page the incident response team for SEV1 breach. Flag `requires_human_review = True` and output information gap codes (`NO_ERROR_DETAILS`, `NO_MONITORING_DATA`) so incident commanders know the report lacks critical telemetry.

#### Scenario B: Pristine Cosmetic SEV4 (Low Severity, High Quality)
- **Input**: 5-page detailed report explaining that a copyright footer on iOS Safari mobile has an 8px margin overlap in landscape mode, complete with DOM node inspection, HAR capture, reproduction steps, and browser version matrices.
- **Severity**: **`SEV4`** (Zero operational downtime, no data risk, zero transactional failure).
- **Report Quality**: **`STRONG` (88.0 / 100)**. Exemplary clarity, reproducibility, and evidence.
- **Triage Action**: File into backlog for next design sprint. `requires_human_review = False`.

### The 5 Quality Dimensions & Strict Weights
Report quality is calculated as a normalized weighted score in $[0.0, 100.0]$:

$$\text{Quality Score} = \left( \frac{\sum_{d \in \text{Dimensions}} W_d \times S_d}{5.0} \right) \times 100.0$$

Where dimension score $S_d \in \{0, 1, 2, 3, 4, 5\}$ and weights $W_d$ sum strictly to $1.00$:

| Dimension | Weight ($W_d$) | Evaluated Criteria |
|:---|:---:|:---|
| **`EVIDENCE_STRENGTH`** | **0.30** | Empirical telemetry: stack traces, Datadog alerts, metric graphs, log timestamps, error rate percentages. |
| **`IMPACT_ARTICULATION`** | **0.25** | Quantified blast radius: % of users impacted, HTTP error rates, degraded tenant IDs, revenue loss. |
| **`CLARITY`** | **0.20** | Unambiguous narrative structure, explicit failure symptoms, technical precision, absence of vague jargon. |
| **`ACTION_CONTEXT`** | **0.15** | Documented diagnostic steps, pod restarts, config rollbacks, failover executions. |
| **`REPRODUCIBILITY`** | **0.10** | Steps to reproduce, preconditions, environment context, trigger vectors. |

#### Quality Bands
- **`POOR`**: $0.0 \le \text{Score} < 35.0$ (Triggers automatic human review)
- **`WEAK`**: $35.0 \le \text{Score} < 55.0$
- **`ADEQUATE`**: $55.0 \le \text{Score} < 75.0$
- **`STRONG`**: $75.0 \le \text{Score} \le 100.0$

---

## 4. Uncertainty Handling & Mathematical Confidence Calibration

### The LLM Overconfidence Problem
When asked to evaluate their own certainty, modern LLMs routinely claim $0.95+$ confidence even when hallucinating or assessing completely unverified one-sentence reports. Relying on self-reported confidence creates dangerous false positives in automated incident triage.

To resolve this, our pipeline calculates **Calibrated Confidence** via deterministic Python arithmetic that caps and penalizes confidence based on empirical evidence and missing data.

### Mathematical Confidence Calibration Formulation

#### 1. Base Model Confidence
Computes the arithmetic mean across the model's self-reported confidence for the 5 quality dimensions and the severity classification:

$$\text{model\_conf} = \frac{1}{6} \left( \sum_{d \in \text{Dimensions}} \text{conf}_d + \text{conf}_{\text{severity}} \right)$$

#### 2. Hard Evidence Ceiling
The presence of concrete telemetry is the ultimate ceiling on confidence. An unverified claim without logs can never exceed $0.30$ confidence:

$$\text{evidence\_ceiling} = 0.30 + (0.14 \times \text{evidence\_score})$$

- $\text{evidence\_score} = 0 \implies \text{evidence\_ceiling} = 0.30$
- $\text{evidence\_score} = 3 \implies \text{evidence\_ceiling} = 0.72$
- $\text{evidence\_score} = 5 \implies \text{evidence\_ceiling} = 1.00$

#### 3. Field Coverage Factor
Rewards reports that populate optional structured fields (`impact`, `evidence`, `actions_taken`):

$$\text{coverage\_ratio} = \frac{\text{provided\_fields\_count}}{\text{total\_optional\_fields}} \quad (\text{where } \text{total\_optional\_fields} = 3)$$

$$\text{coverage\_factor} = 0.75 + (0.25 \times \text{coverage\_ratio})$$

- 0 optional fields provided $\implies 0.75$
- 3 optional fields provided $\implies 1.00$

#### 4. Missing Information Gap Penalty
Penalizes reports for critical information gaps detected by semantic perception:

$$\text{gap\_penalty} = \min(0.25,\, 0.04 \times |\text{missing\_information}|)$$

Detected gap codes include `NO_MONITORING_DATA`, `IMPACT_SCOPE_UNQUANTIFIED`, `NO_ERROR_DETAILS`, `NO_REPRO_STEPS`, `AFFECTED_COMPONENT_UNKNOWN`, `NO_MITIGATION_HISTORY`, `ENVIRONMENT_UNKNOWN`, `NO_TIMELINE`.

#### 5. Clamping & Confidence Banding
The raw confidence is multiplied and bounded within $[0.05, 0.95]$:

$$\text{raw\_conf} = (\text{model\_conf} \times \text{evidence\_ceiling} \times \text{coverage\_factor}) - \text{gap\_penalty}$$

$$\text{confidence} = \text{clamp}(\text{round}(\text{raw\_conf}, 2),\, 0.05,\, 0.95)$$

- **`LOW`**: $\text{confidence} < 0.45$
- **`MEDIUM`**: $0.45 \le \text{confidence} < 0.70$
- **`HIGH`**: $\text{confidence} \ge 0.70$

#### Limiting Reason Attribution
When confidence is degraded, the service exposes the exact bottleneck:
- `WEAK_OR_UNVERIFIED_EVIDENCE`: Triggered when $\text{evidence\_ceiling} < 0.60$.
- `NUMEROUS_INFORMATION_GAPS`: Triggered when $\text{gap\_penalty} \ge 0.15$.

---

### Deterministic Policy Floors
Baseline severity is calculated from an ordinal composite index:

$$\text{Severity Index} = 0.45 \times \text{Scope} + 0.35 \times \text{Criticality} + 0.20 \times \left( \frac{\text{TimeSensitivity}}{3} \times 4 \right)$$

- Baseline: $< 1.0 \implies \text{SEV4}$, $< 2.0 \implies \text{SEV3}$, $< 3.0 \implies \text{SEV2}$, $\ge 3.0 \implies \text{SEV1}$.

Regardless of the baseline index, two deterministic policy floors override severity:
1. **`FLOOR_DATA_LOSS_OR_BREACH_SEV1`**: If business criticality indicates credential exposure, unencrypted PII leakage, or active data loss (`business_criticality >= DATA_LOSS_OR_BREACH`), severity is unconditionally forced to **`SEV1`**.
2. **`FLOOR_TOTAL_OUTAGE_SEV2`**: If impact scope indicates global user impact (`impact_scope >= ALL_USERS`) and core business flows are blocked (`business_criticality >= CORE_FLOW_IMPAIRED`), severity is elevated to at least **`SEV2`**. (Never downgrades a natural SEV1).

---

### Deterministic Human Review Triggers (`requires_human_review = True`)
An incident assessment requires human verification if **any** of the following conditions are met:
1. Calibrated confidence is in the **`LOW`** band ($\text{confidence} < 0.45$).
2. Report quality is in the **`POOR`** band ($\text{score} < 35.0$).
3. Any policy floor was enforced (`FLOOR_DATA_LOSS_OR_BREACH_SEV1` or `FLOOR_TOTAL_OUTAGE_SEV2`).
4. Final incident severity evaluates to **`SEV1`**.
5. The payload was categorized as non-incident or adversarial text (`status == "not_an_incident_report"`).

---

## 5. Property-Based Evaluation & Invariance Testing

Automated evaluation is built into the service repository under [`eval/`](eval/) and functions as a strict quality gate in CI/CD pipelines.

### Test Suites
The test harness runs two complementary suites:
1. **Golden Benchmark Suite ([`eval/cases/golden.json`](eval/cases/golden.json))**: 6 canonical real-world incidents testing SEV1 credential leaks, SEV2 total outages, SEV3 partial degradation, SEV4 cosmetic glitches, vague unstructured Slack messages, and adversarial prompt injection bypass attempts.
2. **Invariance & Monotonicity Suite ([`eval/cases/properties.json`](eval/cases/properties.json))**: Enforces metamorphic and mathematical invariants across incident variations:
   - **`monotonicity_evidence` (Confidence Monotonicity under Evidence Enrichment)**: Adding empirical telemetry (e.g., Datadog error rates, latency spikes) to an identical incident narrative must **never decrease** calibrated confidence:
     $$\text{confidence}_{\text{telemetry}} \ge \text{confidence}_{\text{unverified}}$$
   - **`monotonicity_field_coverage` (Confidence Monotonicity under Field Coverage Expansion)**: Providing structured optional fields (`impact`, `evidence`, `actions_taken`) must **never decrease** calibrated confidence compared to a minimal title/description report:
     $$\text{confidence}_{\text{complete}} \ge \text{confidence}_{\text{minimal}}$$
   - **`breach_policy_floor_invariance` (Severity Policy Floor Invariance under Phrasing Variations)**: Any incident confirming credential leakage must evaluate to SEV1 under `FLOOR_DATA_LOSS_OR_BREACH_SEV1` regardless of whether the phrasing is casual/mild (*"noticed minor config cleanup with root creds exposed"*) or alarmist/urgent (*"CRITICAL EMERGENCY BREACH"*).

### Running the Evaluation Suite
```bash
# Run against offline deterministic mock provider
python -m eval.runner --provider fake

# Run against live OpenAI model
python -m eval.runner --provider openai --model gpt-4o-mini

# Run specific suite
python -m eval.runner --provider fake --suite properties
```

### Sample Evaluation Benchmark Output
```
================================================================================
SUITE 1: GOLDEN BENCHMARK SUITE (golden.json)
================================================================================
STATUS | CASE ID                      |  SEV (EXP/ACT)  |  QUALITY   |  CONFIDENCE  |  REVIEW 
-------------------------------------------------------------------------------------
[PASS] | sev1_credential_leak         |    SEV1/SEV1    |    75.0    | 0.74 (HIGH)  |   YES   
[PASS] | sev2_total_outage_core_flow  |    SEV2/SEV2    |    75.0    | 0.74 (HIGH)  |   YES   
[PASS] | sev3_partial_degradation     |    SEV3/SEV3    |    70.0    | 0.72 (HIGH)  |    NO   
[PASS] | sev4_cosmetic_issue          |    SEV4/SEV4    |    62.0    | 0.58 (MEDIUM) |    NO   
[PASS] | vague_ambiguous_report       |     -/SEV4      |    24.0    |  0.15 (LOW)  |   YES   
[PASS] | adversarial_injection        |     -/SEV4      |    0.0     |  0.05 (LOW)  |   YES   

================================================================================
SUITE 2: INVARIANCE & MONOTONICITY SUITE (properties.json)
================================================================================

Property: [monotonicity_evidence] Confidence Monotonicity under Telemetry Evidence Enrichment
  Description: Adding empirical telemetry and monitoring alerts to an incident report must never decrease calibrated confidence.
  Evidence Monotonicity: weak.conf=0.6700 -> strong.conf=0.7300 (delta: +0.0600)
  Result: [PASS] Assertion satisfied.

Property: [monotonicity_field_coverage] Confidence Monotonicity under Field Coverage Expansion
  Description: Providing optional fields (impact, evidence, actions) must never decrease calibrated confidence compared to minimal title/description input.
  Coverage Monotonicity: minimal.conf=0.1500 -> complete.conf=0.2200 (delta: +0.0700)
  Result: [PASS] Assertion satisfied.

Property: [breach_policy_floor_invariance] Severity Policy Floor Invariance under Phrasing Variations
  Description: Any incident containing confirmed credential exposure must deterministically evaluate to SEV1 regardless of phrasing or tone.
  Variant 'mild_phrasing': severity=SEV1 floors=['FLOOR_DATA_LOSS_OR_BREACH_SEV1']
  Variant 'urgent_phrasing': severity=SEV1 floors=['FLOOR_DATA_LOSS_OR_BREACH_SEV1']
  Result: [PASS] Assertion satisfied.

================================================================================
EVALUATION BENCHMARK SUMMARY REPORT
================================================================================
Provider:            fake
Model:               gpt-4o-mini
Evaluated Suite:     all
--------------------------------------------------------------------------------
Severity Accuracy:          100.0% (4/4)
Policy Floor Enforcement:   100.0% (Required floors: 2)
Total Evaluated Cases:      6
Mean Quality Score:         51.0 / 100.0
Mean Calibrated Confidence: 0.4967
Human Review Flag Rate:     66.7%
Quality Band Breakdown:     {'POOR': 2, 'WEAK': 0, 'ADEQUATE': 2, 'STRONG': 2}
Confidence Band Breakdown:  {'LOW': 2, 'MEDIUM': 1, 'HIGH': 3}
Invariance Property Pass:   100.0% (3/3)
--------------------------------------------------------------------------------
[ALL BENCHMARKS & INVARIANCE PROPERTIES PASSED]
================================================================================
```

---

## 6. Trade-offs & Production Roadmap

### Single-Call Perception vs Multi-Agent Consensus

| Dimension | Single-Call Structured Perception (Current) | Multi-Agent Consensus / Panel of Judges |
|:---|:---|:---|
| **P99 Latency** | **300ms – 800ms** (single LLM round-trip) | 2,500ms – 6,000ms (multiple LLM calls + aggregation) |
| **Token Cost** | ~\$0.0003 per evaluation (gpt-4o-mini) | ~\$0.0015 – \$0.0030 per evaluation ($5\times-10\times$ increase) |
| **Failure Modes** | Upstream provider timeout/rate-limit | Combinatorial failure rate across $N$ parallel judges |
| **Arithmetic Drift** | **Zero** (handled by deterministic Python) | High risk if consensus averages numbers instead of signals |
| **Operational Fit** | **Optimal for active incident triage** where seconds matter during Sev1/Sev2 outages | Better suited for asynchronous, non-time-critical post-mortems |

**Design Rationale**: In production incident response, latency and predictability dominate. By restricting the LLM to a single, strictly conditioned perception call with justification generation before ordinal extraction, and offloading all logic to Python, we achieve the accuracy of consensus panels at a fraction of the latency and cost.

### Production Roadmap & Future Enhancements

1. **Redis Caching by Normalized Content Hash**:
   - Compute SHA-256 digest over normalized title, description, and telemetry.
   - During incident storms (hundreds of duplicate alerts generated simultaneously), serve cached evaluations instantly ($<5\text{ms}$) with zero redundant LLM invocations.
2. **Asynchronous Ingestion Worker (Celery / Redis / ARQ)**:
   - Provide an async ingestion endpoint (`POST /api/v1/assessments/async`) returning a job ticket.
   - Decouple webhook ingress from LLM inference spikes during widespread datacenter outages.
3. **Enterprise Incident Tooling Integrations**:
   - **PagerDuty / Opsgenie**: Webhook listeners that automatically enrich incoming incident pages with report quality feedback and missing information alerts.
   - **Slack / Teams Bot**: Slash-command (`/assess-incident`) providing interactive modals for engineers drafting incident notifications before firing company-wide channels.
   - **Jira / ServiceNow**: Real-time validation gate blocking ticket submission if report quality is `POOR` (< 35.0) unless the submitter overrides with explicit confirmation.
4. **Telemetry & Observability Exporters**:
   - Prometheus metrics endpoint (`/metrics`) exposing counters for evaluated quality bands, severity distribution, policy floor trigger counts, and calibrated confidence histograms.

---

## 7. License & Compliance
This project is licensed under the Apache 2.0 License. Domain logic and schemas adhere to RFC 7807 problem details and strict zero-trust input validation guidelines.
