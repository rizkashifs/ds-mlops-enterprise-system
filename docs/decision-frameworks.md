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
