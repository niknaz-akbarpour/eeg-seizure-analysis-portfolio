"""Reproducible pilot EEG analysis using a small CHB-MIT subset.

Educational/research portfolio code only. Not a medical device.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "edf"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

WINDOW_SECONDS = 5
NEGATIVE_TO_POSITIVE_RATIO = 3
SEIZURE_EXCLUSION_MARGIN_SECONDS = 60
RANDOM_SEED = 20261014

RECORDS = [
    {"subject": "chb01", "file": "chb01_03.edf", "seizures": [(2996, 3036)]},
    {"subject": "chb01", "file": "chb01_04.edf", "seizures": [(1467, 1494)]},
    {"subject": "chb02", "file": "chb02_16.edf", "seizures": [(130, 212)]},
    {"subject": "chb02", "file": "chb02_16+.edf", "seizures": [(2972, 3053)]},
    {"subject": "chb03", "file": "chb03_01.edf", "seizures": [(362, 414)]},
    {"subject": "chb03", "file": "chb03_02.edf", "seizures": [(731, 796)]},
]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}


def overlaps(start: float, end: float, interval_start: float, interval_end: float) -> bool:
    return start < interval_end and end > interval_start


def make_window_starts(duration: float, seizures: list[tuple[int, int]], rng) -> tuple[list[float], list[float]]:
    positive = []
    for seizure_start, seizure_end in seizures:
        positive.extend(
            np.arange(seizure_start, seizure_end - WINDOW_SECONDS + 1e-9, WINDOW_SECONDS).tolist()
        )

    candidates = np.arange(60, duration - WINDOW_SECONDS - 60, WINDOW_SECONDS)
    safe_negative = []
    for start in candidates:
        end = start + WINDOW_SECONDS
        unsafe = any(
            overlaps(
                start,
                end,
                seizure_start - SEIZURE_EXCLUSION_MARGIN_SECONDS,
                seizure_end + SEIZURE_EXCLUSION_MARGIN_SECONDS,
            )
            for seizure_start, seizure_end in seizures
        )
        if not unsafe:
            safe_negative.append(float(start))

    n_negative = min(len(safe_negative), NEGATIVE_TO_POSITIVE_RATIO * len(positive))
    negative = rng.choice(safe_negative, size=n_negative, replace=False).tolist()
    return sorted(positive), sorted(negative)


def preprocess_window(data: np.ndarray, sfreq: float) -> np.ndarray:
    data = data - np.mean(data, axis=-1, keepdims=True)
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def extract_features(data: np.ndarray, sfreq: float) -> dict[str, float]:
    filtered = preprocess_window(data, sfreq)
    eps = np.finfo(float).eps

    rms = np.sqrt(np.mean(filtered**2, axis=-1))
    variance = np.var(filtered, axis=-1)
    line_length = np.mean(np.abs(np.diff(filtered, axis=-1)), axis=-1)
    zero_crossing = np.mean(np.diff(np.signbit(filtered), axis=-1) != 0, axis=-1)
    channel_kurtosis = kurtosis(filtered, axis=-1, fisher=True, bias=False, nan_policy="omit")

    frequencies, psd = welch(
        filtered,
        fs=sfreq,
        nperseg=min(filtered.shape[-1], int(2 * sfreq)),
        axis=-1,
    )
    frequency_mask = (frequencies >= 0.5) & (frequencies <= 40.0)
    total_power = np.trapezoid(psd[:, frequency_mask], frequencies[frequency_mask], axis=-1)

    normalized_psd = psd[:, frequency_mask] / (
        np.sum(psd[:, frequency_mask], axis=-1, keepdims=True) + eps
    )
    spectral_entropy = -np.sum(
        normalized_psd * np.log2(normalized_psd + eps), axis=-1
    ) / np.log2(normalized_psd.shape[-1])

    features = {
        "rms_mean": float(np.mean(rms)),
        "rms_sd_channels": float(np.std(rms)),
        "variance_mean": float(np.mean(variance)),
        "line_length_mean": float(np.mean(line_length)),
        "line_length_sd_channels": float(np.std(line_length)),
        "zero_crossing_mean": float(np.mean(zero_crossing)),
        "kurtosis_mean": float(np.nanmean(channel_kurtosis)),
        "spectral_entropy_mean": float(np.mean(spectral_entropy)),
        "total_power_mean": float(np.mean(total_power)),
    }

    relative_band_power = {}
    for band_name, (low, high) in BANDS.items():
        band_mask = (frequencies >= low) & (frequencies < high)
        absolute = np.trapezoid(psd[:, band_mask], frequencies[band_mask], axis=-1)
        relative = absolute / (total_power + eps)
        relative_band_power[band_name] = float(np.mean(relative))
        features[f"relative_{band_name}"] = relative_band_power[band_name]

    features["theta_alpha_ratio"] = relative_band_power["theta"] / (
        relative_band_power["alpha"] + eps
    )
    features["beta_alpha_ratio"] = relative_band_power["beta"] / (
        relative_band_power["alpha"] + eps
    )
    return features


def build_feature_table() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for record in RECORDS:
        path = DATA_DIR / record["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run download_data.sh first.")

        print(f"Processing {record['file']} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)

        sfreq = float(raw.info["sfreq"])
        duration = raw.n_times / sfreq
        picks = [index for index, name in enumerate(raw.ch_names) if "-" in name]
        positive_starts, negative_starts = make_window_starts(
            duration, record["seizures"], rng
        )

        for label, starts in [(1, positive_starts), (0, negative_starts)]:
            for start in starts:
                first = int(round(start * sfreq))
                last = first + int(round(WINDOW_SECONDS * sfreq))
                data = raw.get_data(picks=picks, start=first, stop=last)
                if data.shape[-1] != int(round(WINDOW_SECONDS * sfreq)):
                    continue
                row = extract_features(data, sfreq)
                row.update(
                    {
                        "subject": record["subject"],
                        "recording": record["file"],
                        "window_start_seconds": start,
                        "label": label,
                        "n_channels": len(picks),
                    }
                )
                rows.append(row)

    features = pd.DataFrame(rows)
    features.to_csv(RESULTS_DIR / "features.csv", index=False)
    return features


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision_pr_auc": average_precision_score(y_true, y_score),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def evaluate_models(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_columns = {
        "subject", "recording", "window_start_seconds", "label", "n_channels"
    }
    feature_columns = [column for column in features.columns if column not in metadata_columns]
    X = features[feature_columns]
    y = features["label"].to_numpy()
    groups = features["subject"].to_numpy()

    models = {
        "Logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    splitter = LeaveOneGroupOut()
    fold_rows = []
    prediction_rows = []
    for model_name, estimator in models.items():
        for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
            model = clone(estimator)
            model.fit(X.iloc[train_index], y[train_index])
            score = model.predict_proba(X.iloc[test_index])[:, 1]
            prediction = (score >= 0.5).astype(int)
            held_out_subject = np.unique(groups[test_index]).item()

            metrics = calculate_metrics(y[test_index], prediction, score)
            metrics.update(
                {
                    "model": model_name,
                    "fold": fold,
                    "held_out_subject": held_out_subject,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                }
            )
            fold_rows.append(metrics)

            for local_index, source_index in enumerate(test_index):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "subject": groups[source_index],
                        "recording": features.iloc[source_index]["recording"],
                        "window_start_seconds": features.iloc[source_index]["window_start_seconds"],
                        "true_label": y[source_index],
                        "predicted_label": prediction[local_index],
                        "predicted_probability": score[local_index],
                    }
                )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary_rows = []
    for model_name, model_predictions in predictions.groupby("model"):
        pooled = calculate_metrics(
            model_predictions["true_label"].to_numpy(),
            model_predictions["predicted_label"].to_numpy(),
            model_predictions["predicted_probability"].to_numpy(),
        )
        pooled.update({"model": model_name, "summary_type": "pooled held-out predictions"})
        summary_rows.append(pooled)

    summary = pd.DataFrame(summary_rows)
    fold_metrics.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "held_out_predictions.csv", index=False)
    summary.to_csv(RESULTS_DIR / "model_summary.csv", index=False)

    final_forest = clone(models["Random forest"]).fit(X, y)
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "random_forest_importance": final_forest.feature_importances_,
        }
    ).sort_values("random_forest_importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "feature_importance_exploratory.csv", index=False)
    return fold_metrics, predictions, summary


def plot_class_counts(features: pd.DataFrame) -> None:
    counts = (
        features.groupby(["subject", "label"]).size().unstack(fill_value=0).rename(columns={0: "Non-seizure", 1: "Seizure"})
    )
    ax = counts.plot(kind="bar", color=["#7fc8d4", "#7b2cbf"], figsize=(8, 5))
    ax.set_title("Five-second EEG windows included in the pilot analysis")
    ax.set_xlabel("Participant")
    ax.set_ylabel("Number of windows")
    ax.legend(title="Annotation")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_counts_by_subject.png", dpi=220)
    plt.close()


def plot_confusion_matrices(predictions: pd.DataFrame) -> None:
    models = predictions["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, models):
        subset = predictions[predictions["model"] == model_name]
        matrix = confusion_matrix(subset["true_label"], subset["predicted_label"], labels=[0, 1])
        image = ax.imshow(matrix, cmap="Purples")
        for (row, column), value in np.ndenumerate(matrix):
            ax.text(column, row, str(value), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Non-seizure", "Seizure"])
        ax.set_yticks([0, 1], ["Non-seizure", "Seizure"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Annotated")
        ax.set_title(model_name)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("Pooled predictions from leave-one-participant-out validation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=220)
    plt.close()


def plot_precision_recall(predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    for model_name, subset in predictions.groupby("model"):
        precision, recall, _ = precision_recall_curve(
            subset["true_label"], subset["predicted_probability"]
        )
        ap = average_precision_score(subset["true_label"], subset["predicted_probability"])
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.2f})")
    prevalence = predictions.drop_duplicates(
        ["subject", "recording", "window_start_seconds"]
    )["true_label"].mean()
    plt.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence={prevalence:.2f}")
    plt.xlabel("Recall (sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision–recall curves: held-out participants")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=220)
    plt.close()


def main() -> None:
    mne.set_log_level("ERROR")
    features = build_feature_table()
    print(f"Created {len(features)} windows from {features['subject'].nunique()} participants.")
    fold_metrics, predictions, summary = evaluate_models(features)
    plot_class_counts(features)
    plot_confusion_matrices(predictions)
    plot_precision_recall(predictions)
    print("\nPooled held-out results:")
    print(
        summary[
            [
                "model", "balanced_accuracy", "sensitivity_recall", "specificity",
                "precision", "f1", "roc_auc", "average_precision_pr_auc"
            ]
        ].round(3).to_string(index=False)
    )
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

