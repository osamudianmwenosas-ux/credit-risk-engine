# Credit Risk Engine — IFRS 9-style educational implementation

An end-to-end, reproducible credit-risk project covering:
- synthetic loan data generation
- Probability of Default (PD)
- Logistic Regression champion vs Gradient Boosting challenger
- calibration and out-of-time validation
- LGD and EAD assumptions
- simplified SICR staging
- scenario-weighted Expected Credit Loss (ECL)
- PSI stability analysis
- automated tests

> Educational project. It is **not** a regulatory IFRS 9 implementation. Thresholds and modelling assumptions are explicitly configurable.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
pytest -q
```

Outputs are written to `outputs/`.

## Methodology

ECL is implemented as:

`ECL = PD × LGD × EAD × discount_factor`

Scenario-weighted ECL is then calculated across base, downside and upside scenarios.

The project deliberately uses synthetic data so the repository can be run without a proprietary banking dataset.
