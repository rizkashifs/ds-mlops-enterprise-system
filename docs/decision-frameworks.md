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
13. [Rules vs. ML vs. LLM: Do You Need a Model at All?](#13-rules-vs-ml-vs-llm-do-you-need-a-model-at-all)
14. [Classification Threshold Selection](#14-classification-threshold-selection)
15. [How Much Labeled Data Do You Need?](#15-how-much-labeled-data-do-you-need)
16. [Handling Class Imbalance](#16-handling-class-imbalance)
17. [Champion-Challenger: When to A/B Test a Model](#17-champion-challenger-when-to-ab-test-a-model)
18. [Handling Label Delay](#18-handling-label-delay)
19. [Human-in-the-Loop vs. Full Automation](#19-human-in-the-loop-vs-full-automation)
20. [Offline Evaluation vs. Online Experiment](#20-offline-evaluation-vs-online-experiment)
21. [Missing Data: Impute, Drop, or Model Through?](#21-missing-data-impute-drop-or-model-through)
22. [CPU vs. GPU for Inference](#22-cpu-vs-gpu-for-inference)

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
---

## 13. Rules vs. ML vs. LLM: Do You Need a Model at All?

Before committing to ML, ask whether a deterministic rule system could solve the problem. Rules are easier to maintain, audit, and explain — and for many business problems, they are good enough.

### Decision tree

```
Can the problem be solved by 3–5 explicit business rules?
  YES → Use a rule engine. No model needed.
  NO  → Is the pattern too complex or high-dimensional for rules?
          YES → Is the input structured/tabular with a label?
                  YES → Traditional ML
                  NO  → Is reasoning or language understanding required?
                          YES → LLM
                          NO  → Define the task more clearly
          NO  → Are the rules changing frequently?
                  YES → Consider ML (rules become a maintenance burden)
                  NO  → Use rules, but document them
```

### Comparison

| Dimension | Rule Engine | Traditional ML | LLM |
|---|---|---|---|
| **Transparency** | Fully explicit | Interpretable (with effort) | Low |
| **Maintenance** | Manual rule updates | Retrain on new data | Prompt updates + retrain |
| **Speed to first value** | Hours | Days–weeks | Hours (with API) |
| **Handles complexity** | Low | High | Very high |
| **Data requirements** | None | Labeled training set | Few-shot or fine-tune |
| **Auditability** | Trivial | Moderate | Hard |
| **Cost** | Near zero | Low–medium | Medium–high at scale |

### When rules win

- Regulatory context requires fully explainable decisions (credit, insurance, medical)
- The decision logic is well-understood and stable
- Edge cases are the exception and can be handled manually
- The business wants to adjust thresholds without redeploying a model

### When to use ML instead of rules

- The signal is in interactions between many features that rules can't capture
- You have labeled historical data showing the right outcome
- Rules are already a complex, brittle mess (50+ conditions)
- The cost of a wrong rule is high (misses too many positives)

### Red flags

- Building an ML model when three IF statements would solve the problem
- Maintaining a 200-rule engine when the same logic could be learned from data
- Using an LLM for a classification task that has 2,000 labeled examples — just train a classifier

---

## 14. Classification Threshold Selection

The default prediction threshold of 0.5 is almost never the right threshold for a production model. The right threshold depends on the cost of false positives vs. false negatives for your specific use case.

### The core tradeoff

```
Low threshold (e.g., 0.2):
  → More positives flagged
  → Higher recall (catch more true positives)
  → Lower precision (more false alarms)
  → Right when missing a true positive is expensive (fraud, disease)

High threshold (e.g., 0.7):
  → Fewer positives flagged
  → Lower recall (miss more true positives)
  → Higher precision (fewer false alarms)
  → Right when acting on a false positive is expensive (manual review, customer friction)
```

### How to choose a threshold

**Step 1: Define the cost asymmetry**

Write out: "For this use case, a false positive costs X and a false negative costs Y."

| Use case | False positive cost | False negative cost | Lean toward |
|---|---|---|---|
| Fraud detection | Low (review friction) | High (fraud loss) | Low threshold |
| Churn intervention | Medium (wasted offer) | Medium (lost customer) | Balanced |
| Loan approval | High (bad debt) | Medium (missed revenue) | High threshold |
| Medical screening | Low (extra test) | Very high (missed diagnosis) | Very low threshold |
| Marketing campaign | Low (ignored email) | Low (missed responder) | Maximise F1 or ROI |

**Step 2: Plot the precision-recall curve**

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
# Plot and pick the threshold that satisfies your precision/recall requirement
```

**Step 3: Use a cost matrix if you can quantify costs**

```python
# Expected value at threshold t:
# EV(t) = TP * value_of_correct_positive
#       - FP * cost_of_false_alarm
#       - FN * cost_of_missed_positive

def expected_value(threshold, y_true, y_prob, tp_value, fp_cost, fn_cost):
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tp * tp_value - fp * fp_cost - fn * fn_cost
```

**Step 4: Fix the threshold at deployment — don't let it float**

Log the chosen threshold as a model parameter in MLflow. The threshold is part of the model definition. Changing it without revalidation is a silent model change.

### What not to do

- Never deploy at 0.5 without checking whether it's the right threshold for your use case
- Never change the threshold in production without logging it and re-evaluating precision/recall
- Never pick a threshold that maximises accuracy — accuracy ignores the cost asymmetry

---

## 15. How Much Labeled Data Do You Need?

The answer depends on the model type, the difficulty of the task, and the class balance. There are no universal rules, but these guidelines prevent teams from either over-investing in labeling or training on too little data.

### Rough minimums by model type

| Model | Minimum viable | Comfortable | Notes |
|---|---|---|---|
| Logistic regression | 500 rows | 5,000+ | Works well with engineered features |
| Random forest | 1,000 rows | 10,000+ | More trees = more data needed |
| Gradient boosting (XGBoost, LightGBM) | 1,000 rows | 10,000+ | Handles noise well |
| Neural network (tabular) | 10,000 rows | 100,000+ | Rarely beats GBM below 50k |
| Fine-tuned LLM | 100–500 examples | 1,000–10,000 | Task-specific; few-shot can work |
| LLM zero-shot | 0 | N/A | No labeling needed; lower ceiling |

These are for binary classification with reasonable class balance. Increase by 3–5× for severe imbalance (< 5% positive rate) or multi-class problems.

### How to check if you have enough: learning curves

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
)
# If validation score is still rising at 100% of data, you need more
# If it has plateaued, you have enough — more data won't help
```

### When you don't have enough labels

| Option | When to use |
|---|---|
| **Weak supervision** (Snorkel, label functions) | Can write heuristic rules; large unlabeled corpus |
| **Active learning** | Label acquisition is expensive; model queries most uncertain samples |
| **Transfer learning / fine-tune** | Related labeled dataset exists in another domain |
| **Synthetic data augmentation** | Structured data; known data-generating process |
| **LLM zero-shot** | Task is language-based; acceptable accuracy without training |
| **Start with rules** | Too few labels for ML; use rules until data accumulates |

### Questions to ask before labeling

1. How are labels being created? Who decides what's correct?
2. What is the inter-annotator agreement? (Low agreement = noisy labels = poor model)
3. Is the labeling definition stable? Changing the definition mid-labeling invalidates prior work.
4. Are labels representative of the full distribution, or only easy cases?

---

## 16. Handling Class Imbalance

Class imbalance is the norm in business ML (fraud: 0.1%, churn: 5–15%, propensity: 10–30%). The wrong response is to do nothing. The second-wrong response is to use SMOTE by default.

### Decision tree

```
What is the positive class rate?

> 20% → No special handling needed; use standard training
10–20% → class_weight='balanced' is usually sufficient
1–10%  → class_weight + adjust decision threshold
< 1%   → Consider undersampling majority + threshold tuning; SMOTE cautiously
```

### Techniques ranked by simplicity and effectiveness

| Technique | How it works | When to use | Risk |
|---|---|---|---|
| **Adjust decision threshold** | Lower threshold from 0.5 to improve recall | First thing to try; no training change | May lower precision |
| **class_weight='balanced'** | Penalise misclassifying minority class more | Easy; often sufficient | Slight overfitting on small datasets |
| **Undersampling majority** | Randomly remove majority class rows | Large majority class; training time matters | Loses information |
| **Oversampling minority (SMOTE)** | Synthesise new minority samples | Minority class too small to learn from | Introduces synthetic data risk; can leak in CV |
| **Two-stage model** | Stage 1 filters; Stage 2 classifies | Extreme imbalance (< 0.1%) | Two models to maintain |

### What to measure

Never use accuracy as the primary metric for imbalanced data. Use:

| Metric | When to use |
|---|---|
| **ROC-AUC** | Ranking models; threshold-independent |
| **PR-AUC (Average Precision)** | Severe imbalance (< 5%); ROC-AUC is misleading |
| **F1 (or F-beta)** | Fixed threshold; care about both precision and recall |
| **Precision at K** | You will act on the top K% only; care about accuracy in that band |

### SMOTE cautions

- Apply SMOTE **after** the train/test split — never before
- Never apply SMOTE to the test set
- SMOTE can inflate CV scores because synthetic samples from the same neighbourhood appear in both train and validation folds if you're not careful
- SMOTE on high-dimensional sparse data rarely helps

---

## 17. Champion-Challenger: When to A/B Test a Model

Not every model upgrade needs a live A/B test. Tests are expensive to run, require traffic splitting, and take time to reach significance. Know when offline evaluation is sufficient and when you genuinely need online measurement.

### Decision tree

```
Does the model output directly drive a customer-facing action?
  NO  → Offline evaluation (holdout) is sufficient
  YES → Is there a clear, measurable business outcome within days/weeks?
          NO  → Offline + shadow mode is sufficient
          YES → Will a 1–2pp improvement in the metric change a business decision?
                  NO  → Offline evaluation is sufficient
                  YES → Run an A/B test
```

### Evaluation modes

| Mode | How it works | When to use |
|---|---|---|
| **Offline holdout** | Evaluate on historical test set | Fast; always required; not sufficient alone for customer-facing decisions |
| **Shadow mode** | New model runs in parallel; results not acted on | Safe validation on live traffic before cutover |
| **Champion-challenger** | X% traffic to new model, (100-X)% to current | Testing a new model with real business outcomes |
| **Canary release** | 1–5% traffic to new model; monitor closely | De-risked rollout; catch production issues early |
| **Full cutover** | Switch 100% traffic to new model | After challenger proves itself or rollback plan is clear |

### What a valid A/B test requires

1. **A measurable business outcome** — not just model AUC, but conversion rate, revenue, incident rate
2. **Statistical significance planning** — calculate sample size before starting, not after
3. **Minimum detectable effect** — what is the smallest lift that matters to the business?
4. **Guard rails** — define what would cause you to stop the test early (safety metrics, degradation)
5. **Fixed duration** — don't stop when results look good; stop at the planned sample size

```python
from scipy import stats

# Minimum sample size per group for a two-proportion test
def min_sample_size(baseline_rate, min_detectable_effect, alpha=0.05, power=0.80):
    from statsmodels.stats.power import zt_ind_solve_power
    effect_size = (min_detectable_effect) / (baseline_rate * (1 - baseline_rate)) ** 0.5
    return int(zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power))
```

### When offline evaluation alone is acceptable

- The model output is not directly customer-facing (internal risk score, batch segment)
- The use case has long feedback loops (outcomes take months to observe)
- The business KPI is not directly attributable to a single model decision
- The improvement over the current model is large (> 10pp AUC) and well-understood

---

## 18. Handling Label Delay

Many real-world ML problems have delayed outcomes: churn happens 30 days after prediction, fraud is confirmed days after the transaction, loan default occurs months after approval. How you handle this delay determines whether your model evaluates and retrains correctly.

### The core problem

```
Prediction date: Jan 1
Outcome observable: Feb 1 (30-day churn window)
If you train on Jan data and evaluate in Jan, you have no labels yet.
If you train on Oct data, you're evaluating a 3-month-old model.
```

### Decision framework

| Delay length | Strategy |
|---|---|
| **< 24 hours** | Near-real-time labels; standard evaluation works |
| **Days–weeks** | Delayed evaluation window; adjust train/test cutoff accordingly |
| **Weeks–months** | Proxy labels + delayed ground truth; two-phase evaluation |
| **> 3 months** | High model staleness risk; consider shorter proxy outcome or shorter horizon |

### Train/test split with label delay

Never use a random split when outcomes are time-lagged. Use a time-based split with a gap equal to the label delay:

```
Timeline:
  Training data:  Jan 2024 → Sep 2024
  Gap:            Sep 2024 → Oct 2024   ← equals the label delay (30 days)
  Test data:      Oct 2024 → Dec 2024   ← outcomes fully observable
  Scoring date:   Jan 2025
```

```python
label_delay_days = 30

train_cutoff = pd.Timestamp("2024-09-01")
test_start   = train_cutoff + pd.Timedelta(days=label_delay_days)
test_cutoff  = pd.Timestamp("2024-12-31")

df_train = df[df["event_date"] < train_cutoff]
df_test  = df[(df["event_date"] >= test_start) & (df["event_date"] < test_cutoff)]
```

### Proxy labels

When the true outcome takes too long, define a proxy outcome observable sooner:

| True outcome | Proxy label | Proxy delay |
|---|---|---|
| Subscription cancellation (30d) | Contacted cancellation team | Same day |
| Loan default (12 months) | First missed payment (30 days) | 30 days |
| Long-term LTV (3 years) | 90-day spend | 90 days |

Proxy labels enable faster retraining cycles. Document the proxy definition and its alignment with the true outcome in the model card.

### Monitoring with label delay

- Do not evaluate model performance until labels are available
- Track the "label coverage rate" (what % of scored records now have labels) daily
- Alert when label coverage drops unexpectedly (upstream data pipeline issue)
- Run performance metrics on a rolling window aligned to label availability

---

## 19. Human-in-the-Loop vs. Full Automation

Not all model decisions should be automated. The right level of human involvement depends on the stakes, reversibility, and regulatory context of the decision.

### Automation levels

| Level | Description | Examples |
|---|---|---|
| **Fully automated** | Model decides; no human review | Spam filtering, ad targeting, recommendation ranking |
| **Human-assisted** (model assists human) | Model surfaces information; human decides | Medical diagnosis aid, legal review, credit analyst dashboard |
| **Human-in-the-loop** (human approves model) | Model proposes; human approves before action | Loan approval, fraud block, large content moderation decisions |
| **Human-on-the-loop** (human monitors model) | Model acts; human monitors and can override | Automated trading guardrails, content policy enforcement |

### Decision framework

```
What is the consequence of an incorrect decision?

Reversible AND low stakes (spam filter, recommendation):
  → Full automation acceptable

Reversible BUT high frequency (ad bidding, search ranking):
  → Full automation with monitoring + kill switch

Irreversible OR high stakes (adverse action, medical, financial):
  → Human-in-the-loop or human-assisted

Regulated decision (credit, insurance, hiring, housing):
  → Human-in-the-loop required by law in most jurisdictions
```

### Risk matrix

| Stakes | Reversibility | Human involvement |
|---|---|---|
| Low | Reversible | Full automation |
| Low | Irreversible | Automated with human-on-the-loop |
| High | Reversible | Human-assisted or human-on-the-loop |
| High | Irreversible | Human-in-the-loop required |

### What to document in the model card

- Automation level (one of the four above)
- Who has override authority
- What triggers a human review flag (score > threshold, edge case, high value)
- Response time SLA for human reviews
- Escalation path for disputed decisions

### Regulatory note

The EU AI Act (2024) and existing sector regulations (FCRA, ECOA, GDPR Art. 22) restrict fully automated decisions in high-stakes contexts. Confirm regulatory requirements with your legal team before deploying any fully-automated adverse action model.

---

## 20. Offline Evaluation vs. Online Experiment

Your holdout AUC does not tell you whether the model improves the business outcome. At some point, you need real-world measurement. The question is when offline is sufficient and when you need to go online.

### When offline evaluation is sufficient

- The model output feeds into a human process (not a fully automated action)
- The business outcome has a very long feedback loop (> 3 months)
- The improvement is large enough (> 10pp) that the business case is clear without an experiment
- The use case is low-stakes (content tagging, internal prioritisation)
- You're replacing a rule engine and can compare directly on historical data

### When you need an online experiment

- The model drives a customer-facing action (send email, approve loan, show ad)
- Small improvements matter (the business is optimising in a narrow performance band)
- You want to measure the causal effect, not just correlation in historical data
- Prior models failed to deliver expected business lift despite good offline metrics

### The intermediate: shadow mode

Run the new model in parallel without acting on its predictions. Compare its output distribution and business-metric predictions to the current model's. This is lower risk than a full A/B test and catches most training-serving skew issues before going live.

```
Shadow mode tells you:
  ✓ The model runs in production without errors
  ✓ Score distributions match expectations
  ✓ Feature computation is consistent
  ✗ Whether the model actually improves the business outcome (need A/B for that)
```

### Online experiment requirements

| Requirement | Detail |
|---|---|
| **Randomisation unit** | Must be the entity being acted on (customer, request) — not day or batch |
| **Sample size** | Pre-calculate based on baseline metric, MDE, and desired power |
| **Duration** | Run for at least 1–2 full business cycles; don't stop early on positive results |
| **Guard rails** | Define metrics that would halt the test (error rate spike, safety metric drop) |
| **Holdout group** | Consider a permanent holdout (5%) that never sees any model change |

---

## 21. Missing Data: Impute, Drop, or Model Through?

Missing data is not a problem to be solved before modelling — it is a signal to be understood first. How you handle missingness should depend on why data is missing, not just how much is missing.

### Types of missingness

| Type | Meaning | Example | Strategy |
|---|---|---|---|
| **MCAR** (Missing Completely At Random) | Missingness is unrelated to any variable | Sensor malfunction; random data entry errors | Safe to drop rows or impute with mean/median |
| **MAR** (Missing At Random) | Missingness depends on observed variables | Older customers less likely to fill in email | Impute using other features (model-based imputation) |
| **MNAR** (Missing Not At Random) | Missingness depends on the missing value itself | High-income customers skip salary field; sick patients skip tests | Treat missingness as a signal; create indicator feature |

### Decision framework

```
Why is data missing?

No meaningful reason (MCAR) AND < 5% missing:
  → Drop rows or mean/median impute; low impact

Depends on other observed features (MAR):
  → Model-based imputation (KNN, regression) or median by group

The missing value IS the signal (MNAR):
  → Create binary "was_missing" feature + fill with placeholder
  → Never discard or impute away this signal

Column missing > 50%:
  → Consider dropping the column entirely
  → Investigate why before assuming imputation is valid
```

### Imputation techniques

| Technique | When to use | Risk |
|---|---|---|
| **Mean / median** | MCAR; numeric; not important feature | Reduces variance; distorts distribution |
| **Mode** | MCAR; categorical | Creates artificial majority |
| **Constant / sentinel** (e.g., -999, "UNKNOWN") | MNAR; tree models handle sentinels well | Model must learn the sentinel's meaning |
| **KNN imputation** | MAR; moderate missingness | Slow at scale; can leak test data |
| **Model-based** (regression imputation) | MAR; important feature; time available | Most accurate; risk of leakage if not careful |
| **Missing indicator** | MNAR; always add alongside imputation | Doubles feature space |

### The training-serving consistency rule

**The most important rule for missing data:** Whatever imputation you apply during training must be applied identically at serving time.

- Fit imputation on the training set only (never on the full dataset)
- Save the imputation parameters (means, mode, model) as artifacts
- Apply the saved parameters at serving time — never recompute

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Fit on training data only
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)

# Save to MLflow
import pickle, mlflow
with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
mlflow.log_artifact("imputer.pkl")

# Apply same imputer at serving time — load from artifact store
```

---

## 22. CPU vs. GPU for Inference

GPU is not always faster or cheaper than CPU for inference. For most traditional ML models and moderate-scale online inference, CPU is the right choice.

### Decision tree

```
Is the model a deep neural network or large language model?
  NO  → CPU is almost always sufficient (sklearn, XGBoost, LightGBM)
  YES → What is the batch size at inference time?
          Small batches (< 32, online inference):
            → CPU may be faster (GPU underutilised with small batches)
          Large batches (> 32, batch inference):
            → GPU likely faster; evaluate cost vs. latency

Is the inference SLA < 50ms?
  YES + DNN/LLM → GPU required
  NO  → CPU may be sufficient
```

### Comparison

| Dimension | CPU | GPU |
|---|---|---|
| **Best for** | Traditional ML, small batches, online inference | Large DNNs, large batch sizes, LLMs |
| **Cost** | Low ($0.02–0.10/hr) | High ($0.50–$5+/hr) |
| **Latency (small batch)** | Often faster (no data-transfer overhead) | Slower for small batches |
| **Latency (large batch)** | Slow (linear) | Fast (parallelised) |
| **Operational complexity** | Low | Medium (driver management, CUDA versions) |
| **Cold start** | Near instant | Seconds (driver init) |

### By model type

| Model | Hardware |
|---|---|
| Logistic regression, Random Forest, XGBoost | CPU always |
| Small neural networks (< 10M params) | CPU for low volume; GPU for large batch |
| Transformer models (BERT, etc.) | GPU for batch; CPU or specialised chip (Inferentia) for online |
| LLMs (7B+ params) | GPU required; consider quantisation (INT8/INT4) |

### Cost optimisation tips

- For batch inference with neural networks: use spot/preemptible GPU instances (50–70% cheaper)
- For LLM inference at scale: evaluate quantised models (INT8 often < 5% quality loss, 2–4× faster)
- For online inference with LLMs: benchmark CPU with ONNX runtime before assuming GPU is needed
- Profile before committing: many teams add GPU instances for sklearn models that run in 1ms on CPU

