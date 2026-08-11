from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import norm

NUMERIC = [
    "annual_income", "loan_amount", "dti", "fico",
    "employment_years", "delinquencies", "interest_rate"
]

def make_data(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", "2024-12-31", periods=n)
    income = np.exp(rng.normal(np.log(55000), 0.55, n))
    fico = np.clip(rng.normal(690, 55, n), 450, 850)
    loan = np.clip(rng.lognormal(np.log(12000), 0.65, n), 1000, 60000)
    dti = np.clip(rng.beta(2.2, 5.0, n) * 0.65, 0.01, 0.9)
    emp = np.clip(rng.normal(6, 4, n), 0, 30)
    delinq = rng.poisson(0.25, n)
    rate = np.clip(0.05 + 0.00008*(700-fico) + rng.normal(0, .008, n), .03, .25)
    z = (
        -2.4
        + 0.000018*(60000-income)
        + 2.0*dti
        + 0.009*(650-fico)
        + 0.18*delinq
        - 0.035*emp
        + 1.2*rate
    )
    pd_true = 1/(1+np.exp(-z))
    default = rng.binomial(1, np.clip(pd_true, .005, .85))
    balance = loan * rng.uniform(.15, 1.0, n)
    recovery = np.clip(rng.normal(.55, .18, n) - .10*default, .05, .95)
    undrawn = np.maximum(loan-balance, 0)
    return pd.DataFrame({
        "date": dates, "annual_income": income, "loan_amount": loan,
        "dti": dti, "fico": fico, "employment_years": emp,
        "delinquencies": delinq, "interest_rate": rate,
        "balance": balance, "recovery": recovery, "undrawn": undrawn,
        "default": default
    })

def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0,1,bins+1)))
    if len(edges) < 3:
        return 0.0
    e, _ = np.histogram(expected, bins=edges)
    a, _ = np.histogram(actual, bins=edges)
    e = np.maximum(e/e.sum(), 1e-6)
    a = np.maximum(a/a.sum(), 1e-6)
    return float(np.sum((a-e)*np.log(a/e)))

def run(n=12000, seed=42, output_dir=Path("outputs")):
    output_dir = Path(output_dir)
    df = make_data(n, seed).sort_values("date").reset_index(drop=True)
    cutoff = df["date"].quantile(.80)
    train = df[df.date < cutoff].copy()
    test = df[df.date >= cutoff].copy()

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC)
    ])
    champion = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, C=0.5))])
    challenger = Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=.06, max_leaf_nodes=15, random_state=seed))])

    Xtr, ytr = train[NUMERIC], train.default
    Xte, yte = test[NUMERIC], test.default
    champion.fit(Xtr, ytr); challenger.fit(Xtr, ytr)
    p1 = champion.predict_proba(Xte)[:,1]
    p2 = challenger.predict_proba(Xte)[:,1]

    metrics = pd.DataFrame([
        ["logistic", roc_auc_score(yte,p1), brier_score_loss(yte,p1), psi(champion.predict_proba(Xtr)[:,1], p1)],
        ["gradient_boosting", roc_auc_score(yte,p2), brier_score_loss(yte,p2), psi(challenger.predict_proba(Xtr)[:,1], p2)]
    ], columns=["model","out_of_time_auc","brier","pd_psi"])
    metrics.to_csv(output_dir/"model_metrics.csv", index=False)

    # Keep the interpretable champion for the downstream ECL example.
    df["pd"] = champion.predict_proba(df[NUMERIC])[:,1]
    df["lgd"] = np.clip(1-df["recovery"], .02, .98)
    df["ead"] = df["balance"] + .25*df["undrawn"]
    df["stage"] = np.select(
        [df["default"].eq(1), df["pd"] >= .18],
        [3, 2], default=1
    )
    df["discount_factor"] = 1/(1+df["interest_rate"])**1
    df["ecl_base"] = df["pd"]*df["lgd"]*df["ead"]*df["discount_factor"]
    df["ecl_downside"] = np.minimum(df["ecl_base"]*1.35, df["ead"])
    df["ecl_upside"] = df["ecl_base"]*.75
    df["ecl"] = .50*df["ecl_base"] + .30*df["ecl_downside"] + .20*df["ecl_upside"]
    df[["date","pd","lgd","ead","stage","ecl"]].to_csv(output_dir/"ecl_results.csv", index=False)

    prob_true, prob_pred = calibration_curve(yte, p1, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted PD"); plt.ylabel("Observed default rate")
    plt.title("PD Calibration — Out-of-Time")
    plt.tight_layout(); plt.savefig(output_dir/"pd_calibration.png", dpi=160); plt.close()

    summary = (
        f"Portfolio ECL: {df.ecl.sum():,.2f}\n"
        f"Stage 1: {(df.stage==1).mean():.1%}\n"
        f"Stage 2: {(df.stage==2).mean():.1%}\n"
        f"Stage 3: {(df.stage==3).mean():.1%}\n"
    )
    (output_dir/"summary.txt").write_text(summary, encoding="utf-8")
    return summary
