# Template: Tabular ML Pipeline

Use this template when you have **structured, tabular data** and a **defined prediction target** (classification or regression).

## When to use this template

- Input data is rows and columns (not text, images, or audio)
- You have a labeled training set
- Output is a score or class prediction
- Decision: see `docs/decision-frameworks.md` §1 (ML vs LLM) and §2 (Batch vs Real-time)

## Setup

1. Copy this template folder to `examples/{your-use-case}/`
2. Update `pipeline.py` — replace all `TODO` items
3. Add your contract to `configs/pipeline_contracts.yaml`
4. Set your thresholds in `configs/training.yaml`
5. Run and verify

## Checklist before first production run

- [ ] Data contract defined and validated (zero violations)
- [ ] Feature engineering code is identical in training and serving
- [ ] Metrics logged: accuracy, F1, ROC-AUC
- [ ] Validation thresholds justified (not just default values)
- [ ] Model card filled in using `docs/model_card_template.md`
- [ ] Failure modes reviewed in `docs/failure-modes.md`
