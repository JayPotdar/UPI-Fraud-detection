# Session 2 - Data Audit

## Completed
- Verified dataset shape (250,000 × 17)
- Checked data types
- No missing values
- No duplicate rows
- Verified transaction_id uniqueness
- Analyzed target distribution
- Identified severe class imbalance (0.192% fraud)

## Key Learnings
- High accuracy does not imply a good fraud detection model.
- Fraud detection datasets are naturally imbalanced.
- Business understanding is essential before model training.

## Next Steps
- Validate categorical values.
- Convert timestamp to datetime.
- Begin data quality validation.

# Shap Analysis
EDA
 ↓
Find potential patterns

Feature Engineering
 ↓
Create meaningful behavioral indicators

Logistic Regression coefficients
 ↓
See global direction of learned relationships

SHAP
 ↓
See how those features contribute to predictions
# SHAP explains the model; it does not validate the model.

# Conclusion
## Final Model Evaluation and Conclusion

### Models Evaluated
The following approaches were evaluated for UPI fraud detection:

- Logistic Regression with class weighting
- Random Forest
- XGBoost with `scale_pos_weight`
- Logistic Regression with SMOTE
- Random Forest with SMOTE
- XGBoost with SMOTE
- Hyperparameter-tuned Logistic Regression
- Threshold tuning

### Key Findings

The dataset is extremely imbalanced, with fraud transactions representing approximately 0.19% of all transactions.

Initial models achieved high overall accuracy in some cases because they predominantly predicted the majority genuine class. Therefore, accuracy alone was not considered an appropriate metric for evaluating fraud detection performance.

Metrics such as:

- Fraud Recall
- Fraud Precision
- F1-score
- PR-AUC
- ROC-AUC
- Confusion Matrix

were used to evaluate the models.

Class weighting improved fraud recall for Logistic Regression but resulted in a large number of false positives.

SMOTE was also evaluated to address class imbalance. However, oversampling did not produce a meaningful improvement in the model's ability to distinguish genuine and fraudulent transactions.

Random Forest and XGBoost also showed limited fraud discrimination despite imbalance-handling techniques.

Hyperparameter tuning was performed for Logistic Regression using Average Precision as the optimization metric. Threshold tuning was subsequently performed to study the trade-off between fraud recall and false positives.

### Final Model

Logistic Regression was retained as the final model because it provided a useful interpretable baseline and allowed detailed analysis of the factors influencing model predictions.

However, the final model showed weak discrimination between genuine and fraudulent transactions. The ROC-AUC remained close to 0.50 and PR-AUC remained very low, indicating that the available features contain limited predictive signal for identifying fraud reliably.

Therefore, the model should not be considered production-ready.

### Explainability

Model behavior was analyzed using:

- Logistic Regression coefficients
- Global SHAP feature importance
- SHAP beeswarm plots
- Local SHAP waterfall explanations

Features such as high-value transaction indicators, age groups, merchant categories, transaction types, network type, and day-of-week influenced model predictions.

A local SHAP analysis also demonstrated an important limitation of the model: a genuine transaction received a relatively high fraud score because several features collectively pushed the prediction toward fraud.

This illustrates why model explainability should be considered alongside predictive performance.

### Final Conclusion

The project demonstrates that handling class imbalance alone is not sufficient for building an effective fraud detection system.

Although techniques such as class weighting, SMOTE, threshold tuning, ensemble models, and hyperparameter optimization were evaluated, the available dataset did not provide enough discriminatory information to reliably separate fraud from genuine transactions.

Future improvements should therefore focus primarily on richer behavioral and historical features such as transaction velocity, customer spending deviation, device history, beneficiary history, account age, geographic anomalies, and historical fraud-risk indicators rather than relying only on further model tuning.
