# Decision Frameworks

> These frameworks answer the questions teams ask before they start building.
> Using them consistently prevents the most common class of ML project failures:
> building the wrong thing with the wrong tool.

---

## Contents

1. [ML vs. LLM: When to use which](#1-ml-vs-llm-when-to-use-which)
2. [Batch vs. Real-Time Inference](#2-batch-vs-real-time-inference)
3. [Retraining Strategy: Time-Based vs. Drift-Based](#3-retraining-strategy-time-based-vs-drift-based)
4. [Build vs. Buy vs. Use a Pre-Built API](#4-build-vs-buy-vs-use-a-pre-built-api)
5. [Simple Model vs. Complex Model](#5-simple-model-vs-complex-model)
6. [Feature Store vs. Pipeline-Computed Features](#6-feature-store-vs-pipeline-computed-features)
7. [Deployment Strategy: Shadow / Canary / A-B / Blue-Green](#7-deployment-strategy-shadow--canary--a-b--blue-green)
8. [Single Model vs. Segmented Models](#8-single-model-vs-segmented-models)
9. [Explainability: When and How Much](#9-explainability-when-and-how-much)
10. [Data Labeling: Manual vs. Programmatic vs. Active Learning](#10-data-labeling-manual-vs-programmatic-vs-active-learning)
11. [Cloud vs. On-Premise vs. Hybrid](#11-cloud-vs-on-premise-vs-hybrid)
12. [When to Retire a Model](#12-when-to-retire-a-model)

---

## 1. ML vs. LLM: When to use which

This is the most commonly misapplied decision in 2024–2026. Teams default to LLMs for everything when a simpler ML model would be faster, cheaper, more auditable, and more robust.

### Decision tree

```
Is the input structured tabular data?
  YES → Use ML (classification, regression, ranking)
  NO  → Is the task reasoning, generation, or language understanding?
          YES → Use LLM
          NO  → Define the task more clearly before choosing
```

### Comparison table

| Dimension | Traditional ML | LLM |
|---|---|---|
| **Input type** | Structured, tabular, numeric | Text, images, documents, unstructured |
| **Output type** | Score, class, numeric prediction | Text, structured extraction, reasoning |
| **Explainability** | High (feature importance, SHAP) | Low (chain-of-thought helps, but limited) |
| **Latency** | <10ms (batch), <100ms (online) | 200ms–5s+ |
| **Cost at scale** | Low ($0.001–0.01 per 1k predictions) | High ($0.10–$10+ per 1k tokens) |
| **Data requirements** | Labeled training set (>1k rows) | Few-shot or zero-shot; large base model |
| **Auditability** | High (reproducible, version-locked) | Harder (non-deterministic by default) |
| **Governance** | Well-understood | Evolving; requires extra care |
| **Drift sensitivity** | Medium (detectable, quantifiable) | High (behavior changes with prompts) |

### When ML is the right choice

- Predicting a numeric outcome from structured features (churn score, LTV, risk rating)
- Ranking or sorting (product recommendations, search relevance)
- Anomaly detection on tabular time series
- Classification with a defined label set and labeled training data
- Any use case where model decisions are subject to regulatory scrutiny

### When LLM is the right choice

- Extracting structured information from free-text (email classification, document parsing)
- Generating summaries, reports, or recommendations from unstructured data
- Conversational interfaces or copilots
- Tasks where labeled training data doesn't exist but good examples do
- Reasoning over heterogeneous inputs (text + tables + images)

### The hybrid case

Most production systems combine both. Classic pattern:

```
Structured features → ML model → score
Unstructured text  → LLM       → extracted features → ML model → final score
```

Example: A churn model uses both account features (ML) and the sentiment of recent support ticket text (LLM extraction → numeric feature → ML).

### Red flags (you may be using the wrong tool)

- Using an LLM to classify something with 5 possible labels and 10,000 labeled examples → just train a classifier
- Using a complex ML model to "understand" free text without any NLP preprocessing → consider an LLM for the text features
- Using an LLM for a real-time decision at scale (millions/day) without a caching or distillation strategy → evaluate cost and latency carefully

---

## 2. Batch vs. Real-Time Inference

The most common mistake is defaulting to real-time when batch would be simpler, cheaper, and more reliable.

### Decision tree

```
Does the consumer need predictions in <1 second?
  YES → Does the prediction depend on data that only exists at request time?
          YES → Real-time inference is required
          NO  → Consider pre-computing scores (batch with cache)
  NO  → Batch inference (scheduled) is the right default
```

### Comparison table

| Dimension | Batch | Real-Time (sync) | Real-Time (async) |
|---|---|---|---|
| **Latency** | Hours to days | <500ms | 1–30s |
| **Infrastructure complexity** | Low | High | Medium |
| **Cost** | Low | High (always-on) | Medium |
| **Freshness** | Stale by design | Real-time | Near real-time |
| **Failure impact** | Delayed; retriable | Customer-facing; cascading | Buffered |
| **Scaling** | Predictable | Spiky; needs autoscaling | Queue absorbs spikes |
| **Debugging** | Easy (logs, files) | Hard (distributed traces) | Medium |

### When to use batch

- Predictions are used in a downstream process that runs on a schedule (nightly, weekly)
- Output is a list, report, or enriched table consumed by another system
- Latency tolerance is >10 minutes
- Examples: churn scoring, LTV calculation, propensity models, risk segmentation

### When to use real-time sync

- Predictions must be served in the same HTTP request/response cycle
- Use case is interactive (user is waiting for a result)
- Freshness is critical (prediction depends on events in the last few seconds)
- Examples: fraud detection on transaction submission, product recommendation on page load

### When to use real-time async

- Predictions need to be fresh but can tolerate seconds of delay
- High volume makes synchronous overhead unacceptable
- Output feeds a downstream process that isn't user-facing
- Examples: post-click scoring, email personalization trigger, alert generation

### The pre-compute pattern (often the best default)

Many "real-time" requirements are actually just "low-latency lookups." If predictions don't depend on data that only exists at request time, pre-compute scores in a batch job and serve them from a fast key-value store.

```
Batch job (every 4h): score all customers → write to Redis/DynamoDB
API call: read(customer_id) → return cached score
```

This gives sub-10ms "real-time" with batch infrastructure cost.

### Cost heuristic

| Prediction volume | Recommended approach |
|---|---|
| <100k/day | Batch (simplest) |
| 100k–10M/day | Batch + cache, or async |
| >10M/day | Needs dedicated architecture review |

---

## 3. Retraining Strategy: Time-Based vs. Drift-Based

### Time-based retraining

Train on a fixed schedule, regardless of drift signals. Simpler to implement and reason about.

**Use when:**
- Business cycles are predictable (seasonal products, quarterly behavior shifts)
- Drift detection infrastructure is not yet in place
- Model performance is stable and retraining risk is low

**Typical cadences:**

| Use case type | Suggested cadence |
|---|---|
| Slowly changing behavior (LTV, risk) | Quarterly |
| Moderately changing behavior (churn, propensity) | Monthly |
| Fast-changing behavior (fraud, pricing) | Weekly or event-driven |
| Real-time environment (ad CTR, recommendations) | Continuous (online learning) |

**Drawback:** You may be retraining unnecessarily, or not retraining fast enough when change accelerates.

### Drift-based retraining

Retrain only when a signal indicates the model is degrading. More efficient but requires monitoring infrastructure.

**Trigger hierarchy (priority order):**

1. **Label drift**: Actual outcomes diverge from predictions by >5pp for 7 consecutive days → immediate retrain
2. **Feature drift**: PSI > 0.20 on any key feature → investigate and likely retrain
3. **Performance degradation**: Model metric (accuracy, F1) drops >5pp below validation baseline → retrain
4. **Upstream event**: Data contract major version bump → mandatory retrain

**Use when:**
- Monitoring infrastructure is in place
- The cost of unnecessary retraining is meaningful
- Model behavior is variable and hard to predict on a schedule

### Hybrid approach (recommended default)

Combine both: set a maximum retraining interval (e.g., quarterly) AND trigger earlier if drift signals fire.

```
Retraining policy:
  - If PSI > 0.20 on any feature → trigger retrain within 1 week
  - If performance drops > 5pp → trigger retrain within 48 hours
  - Otherwise → scheduled monthly retrain
```

### Questions to ask before retraining

1. **Is the drift meaningful or statistical noise?** A PSI of 0.21 on a low-importance feature may not warrant an emergency retrain.
2. **Will fresh data fix the problem?** If the drift is caused by a data pipeline issue (not real-world change), fixing the pipeline is the right action.
3. **Does the new model actually perform better?** A retrained model must be validated — a retrain is not guaranteed to improve things.
4. **Who approves the retrained model?** Define the approval path in advance, especially for automated retraining.

---

## 4. Build vs. Buy vs. Use a Pre-Built API

Before writing any code, ask: does this problem already have a good-enough solution?

| Option | When to choose |
|---|---|
| **Use a pre-built API** (OpenAI, Google Vision, AWS Rekognition) | Generic task (translation, OCR, image classification); speed-to-value > customization |
| **Fine-tune a foundation model** | Task-specific behavior needed; you have labeled data; API latency/cost unacceptable |
| **Train your own model** | Proprietary data advantage; compliance requires on-premise; performance needs exceed APIs |
| **Buy a vendor solution** | Non-core capability; build-and-maintain cost exceeds vendor cost |

**Rule of thumb:** APIs for speed, custom training for competitive advantage, vendor for commodity.

---

## 5. Simple Model vs. Complex Model

The best model is the simplest one that meets the performance threshold.

| Model type | When to use |
|---|---|
| **Logistic regression** | Baseline; explainability required; regulatory scrutiny; features are well-engineered |
| **Random forest / gradient boosting** | Standard default for tabular data; good balance of performance and interpretability |
| **Neural network (tabular)** | Large dataset (>100k rows); many interactions; XGBoost has plateaued |
| **Ensemble / stacking** | Marginal lift needed; production infrastructure can support complexity |

**The 80/20 rule for models:** A well-tuned gradient boosting model typically gets you to 80% of the theoretical maximum performance. If you need the last 20%, be prepared to spend disproportionate time and complexity.

**Complexity costs:**
- Harder to debug
- Slower to retrain
- Harder to explain to stakeholders and regulators
- More likely to overfit or drift unexpectedly

Start simple. Increase complexity only when you can measure the lift and justify the cost.

---

## 6. Feature Store vs. Pipeline-Computed Features

| Approach | When to use |
|---|---|
| **Compute in pipeline** | Features are simple; used by one model; team is small |
| **Centralized feature store** | Features shared across models; consistency between training and serving required; large team |

**Training-serving skew is the #1 failure mode from not using a feature store.** If training computes features differently from inference, the model will degrade in production even if it performs well in evaluation.

Signs you need a feature store:
- More than one model uses the same feature
- You've experienced training-serving skew
- Feature computation is slow and repeated across pipelines
- Different teams are computing the same feature differently

Signs you don't need one yet:
- One model, one team, simple features
- Time-to-value is critical and you can add it later
- You don't have the infrastructure to maintain one

---

## 7. Deployment Strategy: Shadow / Canary / A-B / Blue-Green

Choosing the wrong deployment strategy is how teams ship regressions to 100% of users. Each strategy trades off risk, speed, and observability differently.

### Decision tree

```
Is the new model a major behavioral change (new features, new architecture)?
  YES → Shadow mode first, then canary
  NO  → Is there a measurable business metric to optimize?
          YES → A-B test (requires statistical significance)
          NO  → Canary rollout with monitoring gates
```

### Comparison table

| Strategy | Risk exposure | Observable metric | Rollback speed | When to use |
|---|---|---|---|---|
| **Shadow mode** | Zero (responses discarded) | Divergence from production | N/A | Validating a new model before any live traffic |
| **Canary** | Small % of users | Error rate, latency, business KPI | Fast (redirect traffic) | Incremental rollout with monitoring gates |
| **A-B test** | 50/50 split | Conversion, revenue, click-through | Medium | Measuring true business impact with statistical rigor |
| **Blue-green** | Zero during switch | Post-swap error rate | Instant (DNS flip) | Zero-downtime swap; not for gradual validation |

### Shadow mode

Run the new model against live traffic without serving its predictions. Log both old and new outputs and compare offline.

**Use when:**
- You cannot afford any production risk
- The model is a major change (new architecture, new training data)
- You want to measure prediction divergence before committing

**Exit criteria:** Divergence rate is understood and acceptable, latency is within SLA, no crashes or errors under production load.

### Canary rollout

Route a small percentage of traffic (1–5%) to the new model. Expand in stages if metrics hold.

**Typical rollout gates:**

```
Stage 1:  1% traffic → hold 24h → check error rate, latency, business KPI
Stage 2:  10% traffic → hold 48h → recheck
Stage 3:  50% traffic → hold 48h → recheck
Stage 4: 100% traffic
```

**Automated rollback trigger:** If error rate increases >0.5pp or business KPI drops >2% at any stage, auto-rollback.

### A-B testing

Split users into control (old model) and treatment (new model). Measure a pre-defined business metric until statistical significance is reached.

**Requirements:**
- A clearly defined primary metric (not "model accuracy" — a business outcome)
- Minimum detectable effect defined before the test starts
- Sample size calculated in advance (use a power calculator)
- No peeking: commit to the planned run duration

**Common mistake:** Stopping an A-B test early because the new model "looks better." This inflates false positive rates. Run to completion.

### Blue-green

Maintain two identical production environments. Switch all traffic from blue to green atomically.

**Use when:**
- Schema or API changes require a hard cutover
- You need instant rollback capability without traffic splitting
- Not appropriate for gradual model validation — use canary for that

---

## 8. Single Model vs. Segmented Models

Training one model for all users/segments is simpler. Training separate models per segment can lift performance significantly — but multiplies maintenance burden.

### Decision tree

```
Is there a subgroup where predictions are systematically wrong?
  YES → Is the subgroup large enough to have sufficient training data?
          YES → Consider a segmented model
          NO  → Improve features for that segment instead
  NO  → Is there a regulatory or fairness requirement for per-segment behavior?
          YES → Segmented model or constrained training
          NO  → Single model is sufficient
```

### Comparison table

| Dimension | Single model | Segmented models |
|---|---|---|
| **Training complexity** | Low | High (N training pipelines) |
| **Serving complexity** | Low | Medium (routing layer needed) |
| **Data requirements** | Can pool all data | Each segment needs sufficient volume |
| **Maintenance** | One model to monitor | N models to monitor |
| **Performance** | May underfit high-variance segments | Better per-segment accuracy |
| **Fairness/bias risk** | Higher (single decision boundary) | Controllable per segment |

### When segmentation makes sense

- A known subgroup (product line, geography, user tier) has fundamentally different behavior
- The segment has enough data to train a model that won't overfit
- Business requirements mandate different treatment (regulatory, pricing tiers)
- You've measured that the single model's error rate for the segment is unacceptably high

### When to avoid segmentation

- Segments are small (<5k training rows each) — you'll overfit
- You're solving a data quality problem by segmenting — fix the data instead
- The maintenance cost of N models exceeds the performance gain
- You haven't measured whether the single model is actually underperforming per segment

### The middle path: segment as a feature

Before splitting into separate models, try adding the segment identifier as a feature and letting the model learn the interaction. This often captures most of the lift without the operational cost.

---

## 9. Explainability: When and How Much

Not every model needs to be explainable. Over-investing in explainability for low-stakes models wastes time. Under-investing for regulated decisions is a compliance risk.

### Decision tree

```
Is a human making a consequential decision based on this prediction?
  YES → Is the decision subject to regulatory scrutiny (credit, hiring, insurance)?
          YES → Mandatory explainability; consider intrinsically interpretable models
          NO  → Local explanations (SHAP) sufficient
  NO  → Is the model used internally for triage or prioritization?
          YES → Feature importance at a global level is sufficient
          NO  → No explainability required beyond standard monitoring
```

### Explainability tiers

| Tier | Use case | Approach |
|---|---|---|
| **None** | Internal batch scoring, low-stakes ranking | Model metrics and drift monitoring only |
| **Global (population-level)** | Stakeholder reporting, feature audit | SHAP summary plots, feature importance |
| **Local (per-prediction)** | Analyst investigation, triage support | SHAP waterfall, LIME, counterfactuals |
| **Intrinsic (model is the explanation)** | Regulatory / legal decisions on individuals | Logistic regression, decision tree, scorecard |

### Regulatory baseline (non-negotiable)

For decisions affecting individuals under GDPR Article 22, ECOA, FCRA, or equivalent:
- Individuals have a right to an explanation of automated decisions
- Model must be able to produce adverse action reasons per prediction
- Linear models or scorecards are strongly preferred; black-box models require post-hoc methods with documented accuracy

### Common mistake

Adding SHAP to a black-box model and calling it explainable. SHAP explains the model's behavior, not the real-world causal mechanism. In regulated contexts, this may not satisfy auditors. If regulators require explanation, consider whether a simpler intrinsically interpretable model can meet the performance bar first.

---

## 10. Data Labeling: Manual vs. Programmatic vs. Active Learning

The labeling strategy determines your data flywheel speed, label quality, and cost. Most teams under-invest in this decision.

### Decision tree

```
Do you have ground truth labels from operational outcomes (e.g., payment default, churn)?
  YES → Use those directly (no labeling needed)
  NO  → Is a rule-based heuristic good enough for 80%+ of cases?
          YES → Programmatic labeling (Snorkel / label functions)
          NO  → Do you have budget for human annotation?
                  YES → Manual labeling with quality control
                  NO  → Active learning to maximize label efficiency
```

### Comparison table

| Approach | Label quality | Cost | Speed | Scale |
|---|---|---|---|---|
| **Operational ground truth** | Highest | Zero (already exists) | Fast | Unlimited |
| **Manual annotation** | High (with QC) | High ($0.05–$5/label) | Slow | Limited by headcount |
| **Programmatic (label functions)** | Medium (noisy) | Low | Fast | High |
| **Active learning** | High (selected examples) | Medium | Medium | Efficient |
| **LLM-assisted labeling** | Medium-high | Low-medium | Fast | High |

### Operational ground truth

The best labels are outcomes you already measure. Before designing a labeling process, check:
- Can you observe the outcome within an acceptable time window?
- Is the outcome unambiguous (binary outcome > subjective assessment)?
- Is there selection bias? (You only observe outcomes for users who weren't already filtered.)

### Manual labeling best practices

- Define the label schema and edge cases before annotators start
- Use at least 2 annotators per item; measure inter-annotator agreement (Cohen's kappa > 0.7 is acceptable)
- Label a calibration set first; review disagreements before scaling
- Avoid internal employees as annotators for sensitive decisions

### Programmatic labeling

Write label functions (heuristics, regex, weak classifiers) and combine them using a label model. Tools: Snorkel, Cleanlab.

**Use when:** You can express domain knowledge as rules; you have unlabeled data at scale; manual labeling at full scale is too expensive.

**Risk:** Label noise propagates into the model. Always measure programmatic label accuracy against a small manually-verified holdout.

### LLM-assisted labeling

Use an LLM to generate candidate labels, then have humans verify a sample.

**Use when:** The labeling task requires language understanding that's hard to express as rules; you have a capable foundation model for the domain.

**Risk:** LLM errors can be systematic (not random), creating correlated noise. Measure error rate and bias against ground truth before trusting at scale.

---

## 11. Cloud vs. On-Premise vs. Hybrid

Infrastructure placement affects cost, compliance, latency, and team autonomy. The wrong default adds years of technical debt.

### Decision tree

```
Is there a hard data residency or regulatory requirement (GDPR, HIPAA, FedRAMP)?
  YES → On-premise or private cloud; consult compliance team
  NO  → Does training require hardware you can't provision in the cloud (specialized HPC)?
          YES → On-premise or co-location for training; cloud for serving
          NO  → Cloud-first is the default
```

### Comparison table

| Dimension | Cloud | On-Premise | Hybrid |
|---|---|---|---|
| **Upfront cost** | None | High (CapEx) | Medium |
| **Variable cost** | Per-use (scales with usage) | Fixed (sunk cost) | Mixed |
| **Time to provision** | Minutes | Weeks–months | Varies |
| **Compliance control** | Depends on provider certifications | Full control | Partial |
| **Scaling flexibility** | High | Low | Medium |
| **Vendor dependency** | High | None | Medium |
| **ML tooling ecosystem** | Mature (SageMaker, Vertex, AzureML) | Requires self-managed stack | Mix |

### Cloud-first cases

- New projects where infrastructure requirements are not yet known
- Workloads with spiky compute (experimentation, batch retraining)
- Teams without dedicated infrastructure engineers
- Time-to-value is the top priority

### On-premise cases

- Regulated industries with explicit data residency requirements (healthcare records, financial PII)
- Continuous high-volume inference where reserved hardware is cheaper than on-demand
- Models trained on data that contractually cannot leave the organization's network

### Hybrid pattern (common in enterprise)

Train on-premise (sensitive data stays local) + serve on cloud (low-latency, global edge).

```
Data stays on-premise → training job runs on-prem cluster
Trained artifact (weights, model binary) pushed to cloud model registry
Inference runs on cloud (no raw training data needed at serving time)
```

This satisfies most data residency requirements while retaining cloud serving flexibility.

### Cost trap to avoid

Cloud GPU instances for long-running, predictable training workloads are expensive. If you retrain a large model daily and the schedule is predictable, reserved instances or on-premise dedicated hardware will almost always be cheaper. Run a cost projection at 12 months before committing to on-demand pricing for recurring heavy jobs.

---

## 12. When to Retire a Model

Models are often kept running long past their usefulness, accumulating operational risk, maintenance cost, and organizational confusion. Retirement is a first-class lifecycle decision.

### Retirement triggers

A model should be a retirement candidate when **any** of the following are true:

| Signal | Threshold | Action |
|---|---|---|
| **Performance degradation** | Metric consistently >10pp below launch baseline for 30+ days | Investigate; if unfixable, retire |
| **Business metric decoupling** | Model score no longer correlates with the outcome it was built to predict | Retire immediately |
| **Data source deprecated** | An input feature's upstream source is being shut down | Retrain without it or retire |
| **Use case obsolete** | The downstream process consuming the model is being decommissioned | Retire |
| **Replacement model live** | A successor model is in production and stable for 30+ days | Retire the predecessor |
| **Zero active consumers** | No system or team has queried the model in 90+ days | Retire |

### Retirement checklist

Before decommissioning:

1. **Confirm no consumers.** Check inference logs; contact owners of downstream systems.
2. **Notify stakeholders.** Give 30 days notice for internal models; 60–90 days if external teams depend on it.
3. **Archive artifacts.** Move training code, data snapshots, and model weights to cold storage — don't delete.
4. **Document the retirement.** Record why it was retired in the model registry; include date and owner.
5. **Remove serving infrastructure.** Decommission endpoints, cron jobs, and feature pipelines that exist solely for this model.
6. **Revoke data access.** If the model had dedicated data access (IAM roles, DB credentials), revoke them.

### The shadow-off pattern

Don't hard-cut traffic immediately. Route 0% traffic to the old model for 2 weeks while keeping the serving endpoint live. This gives a rollback path if the successor model has an unforeseen failure. After 2 weeks with no rollback, shut down the endpoint.

### Why teams skip retirement

- "It might be useful again someday" → Archive artifacts; you can always redeploy from the archive
- "I don't know who owns it" → That's a registry hygiene problem; fix it now before retirement gets harder
- "Decommissioning takes more work than leaving it running" → True short-term; not true when the model silently degrades and causes an incident
