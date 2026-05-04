# Git, Versioning, and Release Standards

> This document defines how ML teams version code, manage branches, structure commits,
> and release models and pipelines. These standards exist so that any engineer can
> understand the state of any model at any point in time from the git and MLflow history alone.

---

## Contents

1. [Branching Strategy](#1-branching-strategy)
2. [Commit Standards](#2-commit-standards)
3. [Pull Request Standards](#3-pull-request-standards)
4. [Environment Strategy](#4-environment-strategy)
5. [CI/CD Expectations](#5-cicd-expectations)
6. [Model Versioning Rules](#6-model-versioning-rules)
7. [Release Strategy](#7-release-strategy)
8. [Hotfix Process](#8-hotfix-process)
9. [What Lives in Git vs. MLflow](#9-what-lives-in-git-vs-mlflow)

---

## 1. Branching Strategy

ML repos follow a **trunk-based development** model with short-lived feature branches. Long-lived branches (GitFlow-style `develop` branches) create merge conflicts and make it hard to trace which code produced which model artifact.

### Branch structure

```
master (or main)
  └── Always deployable
  └── Protected — no direct pushes
  └── All merges via reviewed PR

Feature branches (short-lived, < 1 week)
  └── {team}/{use-case}/{short-description}
  └── Examples:
        ds-team/churn/add-support-call-feature
        ml-eng/churn/batch-scoring-service
        platform/infra/mlflow-upgrade
        ds-team/propensity/fix-label-leakage

Release branches (when parallel stabilisation is needed)
  └── release/v{major}.{minor}
  └── Only bug fixes merged in; no new features
  └── Tagged on release

Hotfix branches
  └── hotfix/{model-name}/{short-description}
  └── hotfix/churn-rf/fix-null-feature-crash
  └── Branch from master; merge back to master + current release branch
```

### Rules

- **Never** commit directly to `master`
- Feature branches must be merged within **7 days** of creation — stale branches are deleted
- Branch names must follow the `{team}/{use-case}/{description}` convention
- One logical change per branch — a branch that adds a feature AND refactors a pipeline is two branches

### What counts as a separate branch

| Change type | New branch? |
|---|---|
| New feature / new model experiment | Yes |
| Bug fix | Yes |
| Documentation only | Yes (doc changes still need review) |
| Config change (threshold, param) | Yes — config changes are model changes |
| Dependency version bump | Yes |
| Hotfix in production | Yes — hotfix branch |

---

## 2. Commit Standards

Commits are the audit trail. A well-written commit message tells a reviewer — and a future engineer — what changed, why, and what the effect on the model is.

### Commit message format

```
{type}: {short summary in imperative mood} (≤ 72 chars)

{optional body — explain the WHY, not the WHAT}
{reference to experiment run_id or relevant context}

{optional footer: breaking changes, issue refs}
```

### Commit types

| Type | When to use | Example |
|---|---|---|
| `feat` | New feature or capability | `feat: add support_calls_90d feature to churn pipeline` |
| `fix` | Bug fix | `fix: handle null values in monthly_charges before scoring` |
| `data` | Data contract or schema change | `data: add channel_preference column to propensity contract v1.1` |
| `model` | Model param, threshold, or architecture change | `model: increase n_estimators from 100 to 200 for churn-rf` |
| `config` | Config-only change | `config: lower churn F1 threshold to 0.58 for Q1 retrain` |
| `refactor` | Code restructure with no behaviour change | `refactor: extract feature encoding to shared module` |
| `test` | Add or fix tests | `test: add contract validation tests for null handling` |
| `docs` | Documentation only | `docs: add label delay section to decision-frameworks` |
| `ci` | CI/CD pipeline change | `ci: add model validation gate to PR checks` |
| `chore` | Dependency updates, tooling | `chore: upgrade scikit-learn to 1.4.0` |

### Rules

- Use **imperative mood**: "add feature" not "added feature" or "adds feature"
- The subject line must be ≤ 72 characters
- Reference the MLflow `run_id` in the body when a commit produces or validates a specific training run
- **Never** include secrets, credentials, or data in commits
- **Never** commit Jupyter notebooks with output cells — clear outputs before committing

### Examples

```
model: set class_weight=balanced for churn RandomForest

Default training produced very low recall on churners (< 40%).
Balanced class weights improved recall to 71% with acceptable precision drop.

MLflow run: 174f8900b34e4fada1d7067625648da0
Validation: accuracy=0.81, f1=0.69, roc_auc=0.87 — all above threshold
```

```
fix: prevent training-serving skew in monthly_charges imputation

Training used mean imputation computed on training set.
Scoring pipeline was recomputing the mean on the scoring batch.
Now loads saved imputer artifact from MLflow run at scoring time.

Closes #47
```

---

## 3. Pull Request Standards

Every merge to `master` requires a pull request. PRs are the review gate — they are where code correctness, model correctness, and standards compliance are verified.

### PR title format

```
{type}: {summary} — same format as commit subject line
```

### PR description must include

1. **What changed** — brief description of the change
2. **Why** — business or technical motivation
3. **How to verify** — what the reviewer should check
4. **Model impact** (if applicable) — which model/use case is affected; link to MLflow run
5. **Test evidence** — `pytest tests/ -v` output or CI badge

### PR size limits

| PR type | Maximum lines changed |
|---|---|
| Bug fix | 200 lines |
| Feature | 400 lines |
| Refactor | 600 lines |
| Documentation | No limit |

Large PRs are harder to review and more likely to introduce bugs. If a PR exceeds these limits, split it.

### Reviewer assignment

| Change type | Required reviewers |
|---|---|
| Model params / thresholds | 1 × Data Scientist peer |
| Feature engineering | 1 × Data Scientist peer |
| Data contract change | 1 × Data Engineer |
| Deployment / serving code | 1 × ML Engineer |
| Platform / infrastructure | 1 × Platform Engineer |
| Governance / model card | 1 × Risk/Compliance |

### Before merging

- [ ] All CI checks pass (lint, tests, schema validation)
- [ ] At least one approval from required reviewer
- [ ] No unresolved review comments
- [ ] Branch is up to date with master (rebase or merge)
- [ ] PR description is complete

---

## 4. Environment Strategy

Every model and pipeline runs in one of three environments. Promotion between environments is explicit and gated.

### Environment definitions

| Environment | Purpose | Data | Model stage | Who deploys |
|---|---|---|---|---|
| **Development (dev)** | Experiment, iterate, break things safely | Anonymised sample or synthetic | EXPERIMENTAL | Data Scientist (self-service) |
| **Staging** | Integration testing, pre-production validation | Production-like (real data, access-controlled) | CANDIDATE | ML Engineer via CI/CD |
| **Production (prod)** | Live scoring, real business decisions | Full production data | APPROVED / DEPLOYED | ML Engineer + Platform sign-off |

### Promotion rules

```
dev → staging:
  - All unit tests pass
  - Data contract validation passes on staging data
  - Training pipeline runs end-to-end without errors
  - Validation metrics meet thresholds

staging → production:
  - Staging test suite passes
  - End-to-end integration test with production-like volume passes
  - Pre-deployment checklist complete (see standards/deployment.md)
  - Platform Engineering sign-off
  - Model card approved
```

### Environment configuration

All environment-specific settings (endpoints, credentials, data paths, artifact stores) come from environment variables — never from code or config files checked into git.

```bash
# Dev
export MLOPS_TRACKING_URI=http://localhost:5000
export MLOPS_ARTIFACT_STORE=./artifacts

# Staging
export MLOPS_TRACKING_URI=https://mlflow.staging.internal
export MLOPS_ARTIFACT_STORE=s3://mlops-staging-artifacts

# Production
export MLOPS_TRACKING_URI=https://mlflow.prod.internal
export MLOPS_ARTIFACT_STORE=s3://mlops-prod-artifacts
```

Config files in the repo define **structure** and **defaults** only. They must contain no secrets, no environment-specific endpoints, and no access credentials.

### Data access by environment

| Data type | Dev | Staging | Prod |
|---|---|---|---|
| Production customer data | ❌ Never | ✓ Masked/access-controlled | ✓ Full access, audited |
| Synthetic / anonymised data | ✓ | ✓ | ❌ Not used for scoring |
| Production model artifacts | ❌ | ✓ Read-only | ✓ Read + write (deploy) |

---

## 5. CI/CD Expectations

Every push to a feature branch and every PR to master runs an automated CI pipeline. No merge without a passing CI run.

### Required CI checks (all PRs)

| Check | What it validates | Failure action |
|---|---|---|
| **Linting** (`flake8`, `ruff`) | Code style and syntax | Block merge |
| **Type checking** (`mypy` — optional but recommended) | Type annotations | Warn |
| **Unit tests** (`pytest tests/`) | Correctness of core modules | Block merge |
| **Data contract validation** | All contracts in `configs/pipeline_contracts.yaml` parse and validate | Block merge |
| **Import safety** | No circular imports; all imports resolve | Block merge |

### Additional CI checks (PRs touching model code)

| Check | What it validates | Failure action |
|---|---|---|
| **Training smoke test** | Training pipeline runs on small synthetic dataset | Block merge |
| **Validation gate** | Trained model passes threshold checks | Warn (allow override with justification) |
| **Schema diff** | Data contract changes flagged and reviewed | Require data engineer approval |

### CD pipeline (merge to master)

```
1. Run full test suite
2. Build and tag Docker image (if applicable)
3. Deploy to staging environment
4. Run staging integration tests
5. Await manual approval for production deployment
6. Deploy to production on approval
7. Run post-deployment smoke test
8. Alert on failure
```

### CI/CD configuration

CI configuration lives in `.github/workflows/` (GitHub Actions) or equivalent. CI files are treated like application code — changes require PR review.

### Minimum viable CI (for teams getting started)

If you don't have a full CI/CD pipeline yet, at minimum automate:
1. `pytest tests/` on every push
2. A linting check
3. A mandatory reviewer before merge to master

Add the rest incrementally. Some CI is always better than none.

---

## 6. Model Versioning Rules

Model versions must be unique, monotonically increasing, and semantically meaningful. A version number should tell you what kind of change was made.

### Semantic versioning for models

```
{major}.{minor}.{patch}
   │       │       │
   │       │       └── Hyperparameter tuning only; same architecture and features
   │       └────────── Retrained on fresh data; same architecture; same feature set
   └────────────────── New architecture, new feature set, or breaking change
```

| Change | Version bump | Example |
|---|---|---|
| New algorithm or feature set | Major | `churn-rf:1.0` → `churn-xgb:2.0` |
| Periodic retrain, same pipeline | Minor | `1.0` → `1.1` |
| Hyperparameter tuning only | Patch | `1.0` → `1.0.1` |
| Bug fix in feature computation | Major or Minor | Depends on severity of skew introduced |

### Model naming convention

```
{use-case}-{algorithm}-v{major}

Examples:
  churn-rf-v1
  fraud-xgb-v3
  propensity-gbm-v2
  ltv-linear-v1
```

### MLflow model registry stages

| Stage | Meaning | Who can set |
|---|---|---|
| **None** | Registered but not reviewed | Any engineer |
| **Staging** | In pre-production review/testing | ML Engineer |
| **Production** | Active serving model | Platform Engineer after approval |
| **Archived** | No longer active; retained for audit | Platform Engineer or ML Engineer |

### Version tagging in git

When a model is promoted to Production, tag the git commit that produced it:

```bash
git tag -a model/churn-rf-v1.1 -m "churn-rf v1.1 — promoted to production 2026-05-01"
git push origin model/churn-rf-v1.1
```

This creates a permanent, auditable link between the code and the model artifact.

---

## 7. Release Strategy

A release is a planned promotion of one or more model or pipeline changes into production. Releases are distinct from hotfixes (unplanned urgent changes).

### Release cadence

| Component | Typical cadence | Notes |
|---|---|---|
| Platform / infrastructure | Monthly or on-demand | High coordination required |
| Pipeline code (training, scoring) | Bi-weekly or on-demand | Requires staging validation |
| Model (retrain) | Per retraining trigger or schedule | Follow lifecycle promotion |
| Config changes (thresholds) | On-demand | Treated as a model change; requires review |
| Documentation | On-demand | No deployment needed |

### Release checklist

Before promoting any change to production:

- [ ] All CI checks pass on the release branch
- [ ] Staging deployment validated (pipeline runs, output count correct)
- [ ] Monitoring dashboards reviewed — no pre-existing alerts
- [ ] Rollback procedure documented and tested
- [ ] Consumer teams notified (at least 48h for planned releases)
- [ ] Platform Engineering sign-off obtained
- [ ] Model card updated if model version changed
- [ ] Change record created (change management system)

### Release notes

Every release to production should have release notes documenting:

1. What changed (feature, fix, model version bump)
2. Why (business motivation or technical necessity)
3. Expected impact on scoring output (distribution shift expected? Yes/No)
4. Rollback procedure
5. Who approved

---

## 8. Hotfix Process

A hotfix is an unplanned urgent change to fix a critical production issue. It bypasses the normal release cycle but not the review and testing requirements.

### What qualifies as a hotfix

- Scoring pipeline producing incorrect outputs (data bug, feature bug)
- Model serving failures (errors, crashes, zero-output)
- Security vulnerability in production code
- Data contract violation causing downstream corruption

### Hotfix steps

```
1. Create a hotfix branch from master:
   git checkout master
   git checkout -b hotfix/churn-rf/fix-null-crash

2. Fix the issue (smallest possible change)

3. Write or update the relevant test to cover the bug

4. Get emergency review from at least ONE engineer
   (normally 2 reviewers required; reduced to 1 for hotfix)

5. Merge to master and deploy immediately

6. Tag the hotfix:
   git tag -a hotfix/churn-rf-null-crash-2026-05-01 -m "..."

7. Write a post-incident note within 48 hours:
   - What broke
   - Why it wasn't caught earlier
   - What test or check prevents this in future
```

### What hotfixes are NOT

- Hotfixes are not a fast lane for new features
- A threshold change or model retrain is not a hotfix — follow the normal release process
- If an issue can wait 24 hours without severe business impact, it is not a hotfix

---

## 9. What Lives in Git vs. MLflow

A common source of confusion: which artifacts belong in git and which in MLflow?

| Artifact | Where | Reason |
|---|---|---|
| Pipeline code | Git | Code review, history, CI/CD |
| Config files (params, thresholds) | Git | Reviewable; tied to code version |
| Data contracts | Git | Schema is code; requires review |
| Documentation | Git | Part of the system definition |
| Trained model artifacts (.pkl, .pt) | MLflow artifact store | Large binary; not diff-able |
| Experiment metrics | MLflow | Structured comparison and search |
| Feature importance plots | MLflow | Linked to specific run |
| Imputation parameters | MLflow | Saved per run; loaded at serving time |
| PSI baseline distributions | MLflow | Linked to training run |
| Jupyter notebooks (exploratory) | Git (outputs cleared) | Reproducibility; no binary bloat |
| Raw training data | Neither (object store) | Too large; managed separately |
| Secrets / credentials | Neither (secrets manager) | Never in git or MLflow |

### The rule

**If it changes with every training run → MLflow.**
**If it defines how training runs → Git.**
