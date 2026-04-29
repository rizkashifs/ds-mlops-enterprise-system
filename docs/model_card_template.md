# Model Card: {model_name} v{version}

> Fill in every field. Incomplete model cards block APPROVED status.
> Plain language required — write for a non-technical risk reviewer, not your team.

---

## Overview

| Field | Value |
|---|---|
| **Model name** | {model_name} |
| **Version** | {version} |
| **Owner** | {team} / {name@company.com} |
| **Created date** | {YYYY-MM-DD} |
| **MLflow run ID** | {run_id} |
| **Lifecycle status** | {experimental / candidate / approved / deployed / retired} |

---

## Description

### What this model does
{One paragraph. What does the model predict? What is the output? What is the business process it supports?}

### Intended use
{Specific use cases this model is designed for. Be concrete.}

Example:
> Monthly batch scoring of all active customers (>90 days tenure) to generate a churn probability score. Output feeds the retention marketing team's weekly outreach list.

### Out-of-scope use
{Use cases the model must NOT be used for. Be explicit.}

Example:
> - Real-time scoring for customer service interactions
> - Scoring of prospects or trial customers (not in training distribution)
> - Automated contract termination decisions without human review
> - Any use case involving consumers under 18 years old

---

## Training Data

### Dataset
{Name and version of the data contract used. E.g., `churn_features_v1 v1.0`}

### Date range
{Training data date range. E.g., `2024-01-01 to 2025-12-31`}

### Size
{Number of rows and columns. E.g., `450,000 records × 22 features`}

### Label definition
{How was the target variable defined? What counts as a positive label?}

Example:
> A customer is labeled as churned (target=1) if their account was closed within 30 days of the scoring date.

### Known data issues
{Any known gaps, biases, or limitations in the training data.}

---

## Evaluation

### Test set
{How was the test set constructed? Holdout? Out-of-time split? Cross-validation?}

### Metrics

| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| Accuracy | | | |
| F1 Score | | | |
| ROC-AUC | | | |
| Precision | | | |
| Recall | | | |

### Baseline comparison

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| This model | | | |
| Current production | | | |
| Majority class baseline | | | |

### Fairness analysis

| Group | Accuracy | F1 | Delta vs. overall |
|---|---|---|---|
| {Group A} | | | |
| {Group B} | | | |

{Describe any groups where performance differs by >5pp and how it was addressed.}

---

## Limitations & Risks

### Known limitations
{What is this model bad at? What edge cases will it fail on?}

Example:
> - Underperforms for customers with <3 months tenure (too little signal)
> - Trained before a major pricing change in Q3 2025; may underweight price sensitivity
> - Class imbalance (8% churn rate) means high-probability scores have low precision

### Out-of-distribution risk
{What data patterns would cause this model to fail silently?}

Example:
> - If a new product type is introduced, its customers will have no comparable training examples
> - Geographic expansion to new markets will not be reflected in the training distribution

---

## Ethical Considerations

### Protected attributes
{List any protected attributes that were reviewed. State whether they are in the model or excluded.}

Example:
> The following attributes were explicitly excluded from training features: age, gender, ethnicity, postcode. Monthly income was reviewed as a potential proxy for socioeconomic status.

### Potential for harm
{If the model is wrong, who is affected and how?}

Example:
> False positives result in retention outreach to customers who weren't going to churn (low cost). False negatives mean high-value at-risk customers are not contacted (higher cost). No automated adverse decisions are made based on this model.

### Fairness review outcome
{Was a fairness audit completed? What was the finding?}

---

## Approval Sign-Off

| Reviewer | Role | Date | Decision |
|---|---|---|---|
| | Model Reviewer | | Approved / Rejected |
| | Risk / Compliance | | Approved / Rejected |
| | Platform Engineering | | Approved / Rejected |

### Conditions or caveats
{Any conditions attached to the approval. E.g., "Approved for batch use only; online use requires separate review."}

---

## Change Log

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | {YYYY-MM-DD} | {name} | Initial card |
