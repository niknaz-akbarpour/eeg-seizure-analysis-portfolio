# Analysis Summary

## Dataset constructed

- Participants: 3
- EDF recordings: 6
- Five-second windows: 272
- Annotated seizure windows: 68
- Sampled non-seizure windows: 204
- Validation: leave-one-participant-out (3 folds)

## Pooled held-out performance

| Model | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.833 | 0.809 | 0.858 | 0.655 | 0.724 | 0.884 | 0.589 |
| Random forest | 0.765 | 0.794 | 0.735 | 0.500 | 0.614 | 0.917 | 0.834 |

## Interpretation

Logistic regression produced the stronger threshold-based balance of sensitivity and specificity in this pilot. Random forest ranked windows well by probability, as reflected in its ROC-AUC and PR-AUC, but produced more false-positive classifications at the fixed 0.5 threshold. Performance varied across held-out participants, reinforcing that the pooled values should not be interpreted as evidence of clinical generalization.

## Communication boundary

Use these results to demonstrate a reproducible analytic workflow and understanding of leakage, imbalance, and validation. Do not describe either model as a seizure detector, predictor, safety system, or EpiCircle function.

