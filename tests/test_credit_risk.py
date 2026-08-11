import pandas as pd
from src.credit_risk.pipeline import make_data, psi

def test_synthetic_data_has_required_columns():
    df = make_data(100)
    assert {"fico","dti","loan_amount","default"}.issubset(df.columns)

def test_default_is_binary():
    assert set(make_data(500).default.unique()).issubset({0,1})

def test_psi_non_negative():
    x = pd.Series(range(100))
    assert psi(x, x) >= 0
