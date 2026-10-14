# Pilot EEG Analysis of Annotated Seizure and Non-Seizure Segments

**Niknaz Akbarpour**  
Honours Psychology, Neuroscience & Behaviour (B.Sc.)  
McMaster University

## Project overview

This portfolio project demonstrates a reproducible Python workflow for exploring annotated seizure and non-seizure segments in a small subset of the public CHB–MIT Scalp EEG Database. It was designed to build practical experience with real neurophysiological data, signal processing, feature extraction, participant-grouped validation, and responsible interpretation.

> **Important:** This is an educational, retrospective pilot analysis. It is not a medical device and is not intended to diagnose, predict, monitor, or guide treatment of seizures. It is separate from the current EpiCircle MVP and has not been integrated into that product.

## Research question

Can a small set of interpretable time- and frequency-domain EEG features distinguish annotated seizure windows from sampled non-seizure windows when performance is evaluated on participants excluded from model training?

## Dataset and analytic sample

- Source: CHB–MIT Scalp EEG Database, PhysioNet
- Public, de-identified paediatric scalp-EEG recordings
- Pilot subset: 6 recordings from 3 participants (`chb01`, `chb02`, `chb03`)
- Window length: 5 seconds
- Annotated seizure windows: 68
- Sampled non-seizure windows: 204
- Total windows: 272
- Non-seizure windows were sampled at a planned 3:1 ratio and kept at least 60 seconds away from annotated seizure intervals.

The 25% seizure-window proportion in this constructed analytic sample is **not** an estimate of seizure prevalence in clinical monitoring.

## Workflow

1. Read EDF recordings with MNE-Python.
2. Generate non-overlapping 5-second seizure windows from provided annotations.
3. Sample non-seizure windows from the same recordings outside a 60-second exclusion margin.
4. Apply a 0.5–40 Hz band-pass filter to each window.
5. Extract interpretable aggregate features across EEG channels.
6. Fit logistic regression and random forest baseline classifiers.
7. Evaluate with leave-one-participant-out cross-validation.
8. Report imbalance-aware metrics and limitations.

## Features

- RMS amplitude and between-channel variability
- Variance
- Line length and between-channel variability
- Zero-crossing rate
- Kurtosis
- Spectral entropy
- Total spectral power
- Relative delta, theta, alpha, beta, and low-gamma power
- Theta/alpha and beta/alpha ratios

## Pilot results

All results below pool predictions made for participants who were not included in the corresponding model-training fold.

| Model | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.833 | 0.809 | 0.858 | 0.655 | 0.724 | 0.884 | 0.589 |
| Random forest | 0.765 | 0.794 | 0.735 | 0.500 | 0.614 | 0.917 | 0.834 |

These values describe only this small, deliberately sampled pilot subset. They do not establish clinical validity, real-time performance, prospective generalization, or suitability for deployment.

## Key observation

The exploratory random-forest importance analysis ranked aggregate RMS amplitude, between-channel RMS variation, and line length among the most influential features. Feature importance is descriptive and does not establish physiological causality.

## Validation decision

Random window splitting was not used. Windows from the same participant are highly related and could inflate apparent performance if divided across training and test sets. Instead, leave-one-participant-out validation trained on two participants and tested on the third.

## Repository structure

```text
eeg-seizure-portfolio/
├── README.md
├── METHODS.md
├── LIMITATIONS_AND_ETHICS.md
├── run_analysis.py
├── download_data.sh
├── requirements.txt
├── notebooks/
│   └── EEG_Portfolio_01_Exploration.ipynb
├── figures/
│   ├── class_counts_by_subject.png
│   ├── confusion_matrices.png
│   └── precision_recall_curves.png
└── results/
    ├── features.csv
    ├── fold_metrics.csv
    ├── held_out_predictions.csv
    ├── model_summary.csv
    └── feature_importance_exploratory.csv
```

Raw EDF recordings are excluded from the repository and must be downloaded from PhysioNet.

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash download_data.sh
python run_analysis.py
```

## Data citation

Guttag, J. (2010). *CHB-MIT Scalp EEG Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2K01R

## Suggested CV entry

**Independent EEG Data Analysis Portfolio Project** — Developed a reproducible Python workflow using MNE and scikit-learn to analyse annotated seizure and non-seizure windows from a public scalp-EEG dataset; extracted interpretable signal features and evaluated baseline classifiers using leave-one-participant-out validation, with explicit attention to data leakage, class imbalance, and clinical limitations.

