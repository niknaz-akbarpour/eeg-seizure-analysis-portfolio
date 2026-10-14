# Limitations and Ethical Interpretation

## Appropriate interpretation

This project demonstrates a data-analysis workflow. It does not validate a seizure-detection system. The results may be used to discuss programming, signal-processing, evaluation, and research-design skills, but not to claim clinical capability.

## Major limitations

1. **Very small participant sample:** Only three participants and six recordings were analysed.
2. **Selected rather than representative recordings:** All included recordings contain annotated seizures.
3. **Constructed class balance:** Non-seizure windows were sampled at a 3:1 ratio. This differs substantially from continuous real-world monitoring.
4. **Paediatric, drug-resistant epilepsy context:** Findings may not generalize to adults, community settings, other epilepsy types, or other EEG devices.
5. **Retrospective hospital EEG:** The analysis does not reproduce wearable sensing, real-time processing, motion artefacts, missing electrodes, or home conditions.
6. **Window-level labels:** Windows were labelled from provided onset and end annotations; clinical uncertainty around boundaries was not modelled.
7. **No external validation:** Leave-one-participant-out validation within a three-participant subset is not a substitute for testing on an independent cohort.
8. **No prospective evaluation:** The analysis cannot estimate real-time false-alarm burden, latency, safety, or clinical utility.
9. **No hyperparameter optimization:** Models are intentionally simple baselines.
10. **Feature importance is not causality:** Model importance values do not establish physiological mechanisms.

## Privacy and data governance

The analysis uses a public, de-identified dataset and does not attempt re-identification. Raw EDF data should not be re-hosted in the portfolio repository; users should obtain it from PhysioNet and follow the dataset's terms and citation requirements.

## Clinical and product boundary

This work is separate from the current EpiCircle MVP. The MVP uses simulated data and does not independently detect or predict seizures. No output from this pilot analysis should be presented to users as an alert, diagnosis, risk score, treatment recommendation, or validated monitoring function.

## Required disclosure

> This project is an educational, retrospective analysis of a publicly available, de-identified EEG dataset. It is not a medical device and is not intended to diagnose, predict, monitor, or guide the treatment of seizures. Results should not be generalized to clinical or real-world settings without prospective validation and appropriate regulatory, technical, and clinical oversight.

