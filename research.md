# Credit Risk Research

Research in statistical learning, credit risk modeling, and quantitative decision-making across consumer and commercial lending.

## Research Area 01: Probability of Default

### Candidate Models
- Logistic Regression
- Generalized Additive Models
- Decision Trees
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Survival Models

### Evaluation
- Discrimination (AUC, Gini, KS)
- Calibration (Brier Score, Calibration Slope)
- Population Stability
- Out-of-Time Validation
- Model Interpretability (SHAP)

### Research Question
When do complex machine learning models provide
meaningful improvements over traditional statistical
models in credit risk estimation?




## Research Note 02: Logistic Regression vs. Gradient Boosting

### Objective
Examine the trade-offs between traditional statistical models and gradient-boosted decision trees in Probability of Default (PD) estimation.

### Methodological Considerations
- Logistic regression provides an interpretable baseline with explicit assumptions about the relationship between predictors and log-odds.
- XGBoost and LightGBM capture nonlinear relationships and feature interactions without requiring their manual specification.
- Predictive discrimination should be evaluated using ROC-AUC, Gini, and KS statistics.
- Probability calibration should be assessed independently using calibration curves, Brier scores, and observed-to-expected default ratios.
- Out-of-time validation is essential for assessing robustness to changing borrower populations.

### Research Hypothesis
Gradient-boosted trees may improve discriminatory performance relative to logistic regression, but these improvements do not necessarily translate into superior probability calibration or temporal stability.

### Next Step
Design a controlled benchmark using consistent training, validation, and out-of-time samples across candidate models.




## Research Note 03: Experimental Design for PD Model Benchmarking

### Objective
Establish a consistent experimental framework for comparing statistical and machine learning approaches to Probability of Default (PD) estimation.

### Candidate Models
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Experimental Design
1. Define a consistent default outcome and performance horizon.
2. Partition observations chronologically into training, validation, and out-of-time test samples.
3. Apply preprocessing within training folds to prevent data leakage.
4. Tune hyperparameters using temporal cross-validation where sample history permits.
5. Evaluate all models on identical out-of-time observations.
6. Calibrate predicted probabilities using held-out validation data where necessary.

### Evaluation Criteria
| Dimension | Metrics |
| --- | --- |
| Discrimination | ROC-AUC, Gini, KS, PR-AUC |
| Calibration | Brier Score, Calibration Slope, Observed/Expected Defaults |
| Stability | Out-of-Time Performance, Population Drift |
| Interpretability | Coefficients, SHAP, Permutation Importance |
| Decision Impact | Approval Rate, Expected Loss, Risk-Adjusted Return |

### Research Question
Do improvements in predictive discrimination from ensemble learning translate into economically meaningful gains in lending decisions after accounting for calibration and temporal stability?

### Next Step
Implement a reproducible Python benchmark using a public credit risk dataset.
