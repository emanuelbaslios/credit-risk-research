
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


def evaluate_pd_model(y_true, y_prob, model_name="Model"):
    """Evaluate discrimination and calibration for a binary PD model."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Targets and probabilities must have equal nonzero lengths.")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Targets must contain both binary classes.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    calibration = evaluate_calibration(y, p)
    return pd.DataFrame([{
        "Model": model_name,
        "N": len(y),
        "Default Rate": y.mean(),
        "ROC-AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": np.max(tpr - fpr),
        "Brier Score": brier_score_loss(y, p),
        "Log Loss": log_loss(y, p, labels=[0, 1]),
        **calibration
    }])

def evaluate_calibration(y_true, y_prob):
    """Evaluate PD calibration using logistic recalibration."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Targets and probabilities must have equal nonzero lengths.")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Targets must contain both binary classes.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    if np.unique(p).size < 2:
        raise ValueError("Calibration slope requires varying predictions.")

    p_safe = np.clip(p, 1e-6, 1 - 1e-6)
    log_odds = logit(p_safe).reshape(-1, 1)

    model = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=2000
    )
    model.fit(log_odds, y)

    intercept_model = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=2000
    )
    offset = log_odds.ravel()
    from scipy.optimize import minimize_scalar
    from scipy.special import expit
    from sklearn.metrics import log_loss

    fit = minimize_scalar(
        lambda a: log_loss(y, expit(a + offset), labels=[0, 1]),
        bounds=(-20, 20), method="bounded"
    )

    return {
        "Calibration Intercept": fit.x,
        "Calibration Slope": model.coef_[0, 0],
        "O/E Ratio": y.sum() / p.sum() if p.sum() > 0 else np.nan
    }

def compare_pd_models(y_true, predictions):
    """Compare models using the same observed outcomes."""
    return pd.concat([
        evaluate_pd_model(y_true, prob, name)
        for name, prob in predictions.items()
    ], ignore_index=True).sort_values("ROC-AUC", ascending=False)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=3000, n_features=12, n_informative=6,
        weights=[0.85, 0.15], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, random_state=42
        )
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        )
    except ImportError:
        print("XGBoost not installed; skipping XGBoost model.")

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict_proba(X_test)[:, 1]
    print(compare_pd_models(y_test, predictions).round(4).to_string(index=False))
