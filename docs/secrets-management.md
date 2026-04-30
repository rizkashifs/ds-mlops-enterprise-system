# Secrets and Credentials Management

> The rule is simple: secrets never appear in source code, config files committed to the repository, or log output. Violating this rule creates breaches that are expensive to remediate and impossible to fully undo.

---

## Contents

1. [The non-negotiables](#1-the-non-negotiables)
2. [How secrets flow into the application](#2-how-secrets-flow-into-the-application)
3. [Using src/core/secrets.py](#3-using-srccoresecretsspy)
4. [Local development setup](#4-local-development-setup)
5. [CI/CD secrets](#5-cicd-secrets)
6. [Cloud secrets managers](#6-cloud-secrets-managers)
7. [Credential rotation](#7-credential-rotation)
8. [What counts as a secret](#8-what-counts-as-a-secret)
9. [Incident response: leaked secret](#9-incident-response-leaked-secret)

---

## 1. The non-negotiables

These apply without exception:

- **No secrets in Python files.** Not hardcoded, not as default argument values.
- **No secrets in YAML config files** committed to the repository.
- **No secrets in `.env` files committed to the repository.** Add `.env` to `.gitignore` — it is already there.
- **No secrets in log output.** Never log a secret, even partially masked (last 4 chars visible is still a leak in structured logs).
- **No secrets in Jupyter notebooks.** Notebooks are frequently shared and committed. Load secrets the same way as any other Python file.
- **No secrets passed as command-line arguments.** They appear in `ps aux` and shell history.

If you find a secret in any of these places: treat it as compromised. Rotate it immediately. Then fix the code.

---

## 2. How secrets flow into the application

The only approved pattern is: **secret store → environment variable → application**.

```
Secret store                    Container / process               Application
(AWS Secrets Manager,    ──►    Environment variable      ──►    src/core/secrets.py
 HashiCorp Vault,               MY_SECRET=<value>                require("MY_SECRET")
 GCP Secret Manager,
 GitHub Actions secrets,
 k8s Secret)
```

The application code never knows or cares where the secret came from — only that the environment variable exists.

---

## 3. Using src/core/secrets.py

The `src/core/secrets.py` module provides three functions:

### require(name) — use for mandatory secrets

```python
from src.core.secrets import require

db_password = require("DB_PASSWORD")        # raises EnvironmentError if missing
mlflow_token = require("MLFLOW_TRACKING_TOKEN")
```

Call `require()` at application startup (module level or in an `__init__` function), not lazily inside request handlers. This ensures the application fails fast with a clear error if a secret is missing, rather than failing on the first request an hour after deployment.

### get(name, default) — use for optional configuration

```python
from src.core.secrets import get

log_level = get("LOG_LEVEL", default="INFO")
```

### require_all(*names) — validate all at once

```python
from src.core.secrets import require_all

creds = require_all("DB_HOST", "DB_USER", "DB_PASSWORD")
# raises EnvironmentError listing ALL missing names at once
```

---

## 4. Local development setup

Create a `.env` file in the project root (already in `.gitignore`):

```bash
# .env  — never commit this file
MLFLOW_TRACKING_URI=http://localhost:5000
DB_PASSWORD=localdevpassword
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Load it before starting your process. Options:

**Option A — shell export (simplest)**
```bash
export $(grep -v '^#' .env | xargs)
python pipelines/training_pipeline/train.py
```

**Option B — python-dotenv in a dev entrypoint (not in production code)**
```python
# dev_entrypoint.py — do not import from production modules
from dotenv import load_dotenv
load_dotenv()
# ... rest of dev startup
```

**Option C — direnv (automatically loads .env on cd)**
```bash
brew install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
# .envrc in project root:
dotenv
```

Never import `dotenv` from production pipeline or service code. If it runs, the environment should already be set.

---

## 5. CI/CD secrets

### GitHub Actions

Store secrets in **Settings → Secrets and variables → Actions**. Reference them in workflows:

```yaml
- name: Run tests
  run: pytest tests/ -v
  env:
    MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
    DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
```

Rules:
- Create one secret per value. Do not bundle multiple secrets into a single JSON blob.
- Secrets needed only for integration tests belong in a separate environment (Settings → Environments) with restricted access.
- Rotate CI secrets on the same schedule as production secrets.

### What the CI pipeline should NOT need

Unit tests should not require real credentials. If a test requires a live MLflow server, a real database, or a cloud bucket, it is an integration test — mock it or move it to a separate pipeline.

The `ci.yml` in this repo runs unit tests only. `MLFLOW_TRACKING_URI` is intentionally set to `""` so that any accidental MLflow network calls fail immediately rather than hanging.

---

## 6. Cloud secrets managers

### AWS Secrets Manager

```python
import boto3, json
from src.core.secrets import require

def load_db_creds() -> dict:
    secret_name = require("DB_SECRET_ARN")
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])
```

At container start, a sidecar or init container can inject the values as environment variables using the AWS Secrets Manager agent — the application then reads them via `require()` with no AWS SDK dependency in application code.

### HashiCorp Vault

Use the Vault Agent sidecar pattern: Vault Agent writes secrets to a temp file or injects them as environment variables at container start. Application code reads env vars; it never contacts Vault directly.

### Kubernetes Secrets

Mount secrets as environment variables, not as files, unless a file format is specifically required (e.g., TLS cert). Files on disk can be accidentally included in logs, container images, or copied artifacts.

```yaml
env:
  - name: DB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: db-credentials
        key: password
```

---

## 7. Credential rotation

| Credential type | Rotation cadence | Automated? |
|---|---|---|
| Service account API keys | Every 90 days | Yes (AWS Secrets Manager auto-rotation) |
| Database passwords | Every 90 days | Yes |
| MLflow tracking token | Every 180 days | Manual |
| Cloud storage access keys | Every 90 days | Yes (IAM roles preferred; avoid long-lived keys) |
| CI/CD secrets (GitHub Actions) | Every 90 days | Manual |
| Developer local credentials | On team member departure | Manual |

**Prefer short-lived credentials over long-lived ones.** IAM roles (AWS), Workload Identity (GCP), and Managed Identities (Azure) grant temporary credentials without any stored secret — use them for compute resources whenever possible.

After rotation:
1. Update the secret in the secret store.
2. Restart or redeploy services that cache the old value at startup.
3. Verify the service starts cleanly with `require()` catching any misses.
4. Revoke the old credential after confirming the new one works.

---

## 8. What counts as a secret

| Is a secret | Not a secret |
|---|---|
| Passwords, passphrases | Usernames (unless they reveal account structure) |
| API keys and tokens | Public endpoint URLs |
| Database connection strings with credentials | Database hostnames (without creds) |
| Private TLS certificates and private keys | Public TLS certificates |
| AWS access key + secret key pairs | AWS region |
| OAuth client secrets | OAuth client IDs |
| Encryption keys and salts | Algorithm names |
| Personal access tokens (GitHub, MLflow, etc.) | Repository names, model names |
| Any value labelled `_SECRET`, `_KEY`, `_TOKEN`, `_PASSWORD` | Feature flags, thresholds, config values |

When in doubt, treat it as a secret.

---

## 9. Incident response: leaked secret

If a secret is committed to the repository or otherwise exposed:

1. **Rotate immediately.** Do not wait to assess scope. Rotate the credential first.
2. **Revoke the old credential.** After the new one is confirmed working, revoke the old one — don't just stop using it.
3. **Purge from git history.** A secret in a commit is visible to anyone with repo access, now and forever, unless history is rewritten.
   ```bash
   git filter-repo --path-regex '.*' --replace-text <(echo "oldvalue==>REDACTED")
   git push origin --force --all
   ```
   Note: this requires force-pushing; coordinate with the team. All clones are invalidated.
4. **Audit access logs.** Check cloud provider logs for use of the exposed credential from unexpected sources.
5. **Document the incident.** Date, what was exposed, for how long, what was rotated, what (if anything) was accessed. This is required for SOC2 and most compliance frameworks.
6. **Add a pre-commit hook** to prevent recurrence:
   ```bash
   pip install detect-secrets
   detect-secrets scan > .secrets.baseline
   # add to .pre-commit-config.yaml
   ```
