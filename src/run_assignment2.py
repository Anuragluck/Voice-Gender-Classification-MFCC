"""EC24313 Assessment 2: gender classification from voice recordings.

Student: Anurag Tiwari
Roll Number: BTECH/10142/24
Department of Electronics and Communication Engineering
Birla Institute of Technology Mesra, Ranchi
GitHub Repository: https://github.com/Anuragluck/Voice-Gender-Classification-MFCC

This script extracts 13 MFCC coefficients from each usable recording, trains a
class-balanced SVM, and writes all required figures and evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SAMPLE_RATE = 16_000
N_MFCC = 13
TOP_DB = 30
MIN_DURATION_SECONDS = 0.25
RANDOM_STATE = 42
SVM_C = 4.0
CLASS_TO_LABEL = {"boys": 1, "girls": 2}
FOLDER_TO_CLASS = {"VOICE DATASET BOYS": "boys", "VOICE DATASET GIRLS": "girls"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run MFCC extraction, SVM training, and evaluation for Assessment 2."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project directory containing data/raw/VOICE DATASET.zip.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def prepare_dataset(project_root: Path) -> Path:
    """Extract the supplied archive once and return the dataset directory."""
    archive = project_root / "data" / "raw" / "VOICE DATASET.zip"
    extract_root = project_root / "data" / "extracted"
    expected = extract_root / "VOICE DATASET BOYS"
    if not archive.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    if not expected.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as dataset_zip:
            dataset_zip.extractall(extract_root)
    return extract_root


def safe_error(error: Exception | str) -> str:
    return str(error).replace("\n", " ").strip()[:500]


def speaker_group_id(path: Path) -> str:
    """Create a conservative speaker group from inconsistent file names.

    The dataset has paired BIT/name utterances for many students. Grouping by the
    first name-like token keeps those related files in the same CV fold and avoids
    train/test speaker leakage. The generated manifest exposes every group for review.
    """
    text = path.stem.lower().replace("&", " ")
    text = re.sub(r"[_.()\-]+", " ", text)
    prefix = re.split(r"\b(?:bit|name)\b", text, maxsplit=1)[0]
    tokens = re.findall(r"[a-z]+", prefix)
    ignored = {"audio", "processed", "recording", "voice", "raw", "trimmed", "wav", "aiml"}
    tokens = [token for token in tokens if token not in ignored]
    if tokens:
        return tokens[0]
    fallback = re.findall(r"[a-z]+", text)
    return fallback[0] if fallback else f"unresolved_{path.name.lower()}"


def decode_with_ffmpeg(audio_path: Path, sample_rate: int) -> np.ndarray:
    """Fallback decoder for MP4/M4A and unusual extensions."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(message or "FFmpeg could not decode the file")
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Decoded audio is empty")
    return audio


def load_audio(audio_path: Path) -> np.ndarray:
    """Decode, mono-convert, resample, normalize, and trim edge silence."""
    try:
        audio, source_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if source_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=source_rate, target_sr=SAMPLE_RATE)
    except Exception:
        audio = decode_with_ffmpeg(audio_path, SAMPLE_RATE)

    audio = np.asarray(audio, dtype=np.float32)
    audio = audio[np.isfinite(audio)]
    if audio.size == 0:
        raise ValueError("Audio contains no finite samples")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    trimmed, _ = librosa.effects.trim(audio, top_db=TOP_DB)
    if trimmed.size / SAMPLE_RATE < MIN_DURATION_SECONDS:
        raise ValueError(
            f"Usable audio after trimming is shorter than {MIN_DURATION_SECONDS:.2f} seconds"
        )
    return trimmed


def extract_features(audio: np.ndarray) -> dict[str, float]:
    """Summarize the 13 frame-level MFCC coefficients into fixed-size features."""
    coefficients = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=256,
    )
    means = coefficients.mean(axis=1)
    stds = coefficients.std(axis=1)
    features: dict[str, float] = {}
    for index, value in enumerate(means, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(value)
    for index, value in enumerate(stds, start=1):
        features[f"mfcc_{index:02d}_std"] = float(value)
    return features


def build_feature_table(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return successful feature rows and a full manifest including failures."""
    feature_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder = dataset_root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Expected class folder is missing: {folder}")
        for audio_path in sorted(path for path in folder.rglob("*") if path.is_file()):
            relative_path = audio_path.relative_to(dataset_root).as_posix()
            base_row = {
                "relative_path": relative_path,
                "filename": audio_path.name,
                "class_name": class_name,
                "label": CLASS_TO_LABEL[class_name],
                "group_id": speaker_group_id(audio_path),
                "extension": audio_path.suffix.lower() or "[none]",
            }
            try:
                audio = load_audio(audio_path)
                features = extract_features(audio)
                row = {
                    **base_row,
                    "trimmed_duration_seconds": round(len(audio) / SAMPLE_RATE, 4),
                    **features,
                }
                feature_rows.append(row)
                manifest_rows.append({**base_row, "status": "included", "reason": ""})
            except Exception as error:
                manifest_rows.append({**base_row, "status": "excluded", "reason": safe_error(error)})

    features = pd.DataFrame(feature_rows)
    manifest = pd.DataFrame(manifest_rows)
    if features.empty:
        raise RuntimeError("No audio files could be decoded into usable MFCC features")
    return features, manifest


def plot_class_distribution(features: pd.DataFrame, output_path: Path) -> None:
    counts = features["class_name"].value_counts().reindex(["boys", "girls"], fill_value=0)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(["Boys (Class 1)", "Girls (Class 2)"], counts.values, color=["#2f6f9f", "#c05b83"])
    axis.set_title("Usable recordings by class")
    axis.set_ylabel("Number of recordings")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, counts.values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(features: pd.DataFrame, feature_columns: list[str], output_path: Path) -> None:
    scaled = StandardScaler().fit_transform(features[feature_columns])
    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    pca_frame = pd.DataFrame(
        {"PC1": coordinates[:, 0], "PC2": coordinates[:, 1], "Class": features["class_name"].str.title()}
    )
    fig, axis = plt.subplots(figsize=(7.4, 5.5))
    sns.scatterplot(
        data=pca_frame,
        x="PC1",
        y="PC2",
        hue="Class",
        style="Class",
        palette={"Boys": "#2f6f9f", "Girls": "#c05b83"},
        s=65,
        alpha=0.82,
        ax=axis,
    )
    axis.set_title("PCA scatter plot of 13-MFCC summary features")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_direct_mfcc_scatter(features: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 5.5))
    sns.scatterplot(
        data=features,
        x="mfcc_02_mean",
        y="mfcc_03_mean",
        hue="class_name",
        style="class_name",
        palette={"boys": "#2f6f9f", "girls": "#c05b83"},
        s=65,
        alpha=0.85,
        ax=axis,
    )
    axis.set_title("Acoustic Feature Space: MFCC 2 Mean vs MFCC 3 Mean")
    axis.set_xlabel("MFCC Coefficient 2 Mean (Spectral Tilt / Low Formants)")
    axis.set_ylabel("MFCC Coefficient 3 Mean")
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Boys (1)", "Girls (2)"],
        yticklabels=["Boys (1)", "Girls (2)"],
        ax=axis,
    )
    axis.set_title("Out-of-fold confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, girls_scores: np.ndarray, auc_score: float, output_path: Path) -> None:
    binary_truth = (y_true == CLASS_TO_LABEL["girls"]).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(binary_truth, girls_scores)
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.plot(false_positive_rate, true_positive_rate, color="#c05b83", lw=2.3, label=f"SVM (AUC = {auc_score:.3f})")
    axis.plot([0, 1], [0, 1], color="#5f6368", linestyle="--", lw=1.2, label="Chance")
    axis.set_title("ROC curve for girls as the positive class")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate (sensitivity)")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def choose_folds(features: pd.DataFrame) -> int:
    groups_per_class = features.groupby("class_name")["group_id"].nunique()
    smallest_group_count = int(groups_per_class.min())
    if smallest_group_count < 2:
        raise ValueError("At least two unique speaker groups are needed in each class for evaluation")
    return min(5, smallest_group_count)


def evaluate_model(
    features: pd.DataFrame, feature_columns: list[str], seed: int
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    x_values = features[feature_columns].to_numpy(dtype=float)
    y_values = features["label"].to_numpy(dtype=int)
    groups = features["group_id"].to_numpy()
    folds = choose_folds(features)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.zeros(len(features), dtype=int)
    girls_scores = np.zeros(len(features), dtype=float)
    fold_numbers = np.zeros(len(features), dtype=int)

    for fold_number, (train_index, test_index) in enumerate(splitter.split(x_values, y_values, groups), start=1):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(kernel="rbf", C=SVM_C, gamma="scale", class_weight="balanced", random_state=seed),
                ),
            ]
        )
        model.fit(x_values[train_index], y_values[train_index])
        predictions[test_index] = model.predict(x_values[test_index])
        decision_scores = model.decision_function(x_values[test_index])
        classes = model.named_steps["svm"].classes_
        girls_index = int(np.where(classes == CLASS_TO_LABEL["girls"])[0][0])
        girls_scores[test_index] = decision_scores[:, girls_index] if decision_scores.ndim == 2 else decision_scores
        fold_numbers[test_index] = fold_number

    matrix = confusion_matrix(y_values, predictions, labels=[1, 2])
    metrics = {
        "model": f"RBF SVM (C={SVM_C}) with StandardScaler and class_weight=balanced",
        "sample_rate_hz": SAMPLE_RATE,
        "silence_trim_threshold_db": TOP_DB,
        "mfcc_coefficients": N_MFCC,
        "model_feature_count": len(feature_columns),
        "evaluation": "Stratified group k-fold cross-validation using filename-derived speaker groups",
        "folds": folds,
        "positive_class_for_sensitivity_precision_roc": "girls (Class 2)",
        "accuracy": float(accuracy_score(y_values, predictions)),
        "sensitivity_recall_girls": float(recall_score(y_values, predictions, pos_label=2, zero_division=0)),
        "precision_girls": float(precision_score(y_values, predictions, pos_label=2, zero_division=0)),
        "roc_auc_girls": float(roc_auc_score((y_values == 2).astype(int), girls_scores)),
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "classification_report": classification_report(
            y_values,
            predictions,
            labels=[1, 2],
            target_names=["boys", "girls"],
            output_dict=True,
            zero_division=0,
        ),
    }
    prediction_table = features[
        ["relative_path", "filename", "class_name", "label", "group_id", "trimmed_duration_seconds"]
    ].copy()
    prediction_table["fold"] = fold_numbers
    prediction_table["predicted_label"] = predictions
    prediction_table["predicted_class"] = np.where(predictions == 1, "boys", "girls")
    prediction_table["girls_decision_score"] = girls_scores

    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=SVM_C, gamma="scale", class_weight="balanced", random_state=seed)),
        ]
    )
    final_model.fit(x_values, y_values)
    return metrics, prediction_table, final_model


def evaluate_model_benchmarks(
    features: pd.DataFrame, feature_columns: list[str], seed: int
) -> pd.DataFrame:
    """Benchmark multiple classifier families (SVM, RF, MLP, LR) under identical group-CV."""
    x_values = features[feature_columns].to_numpy(dtype=float)
    y_values = features["label"].to_numpy(dtype=int)
    groups = features["group_id"].to_numpy()
    folds = choose_folds(features)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    candidate_models = {
        "SVM (RBF, C=4.0, balanced)": Pipeline(
            [("scaler", StandardScaler()), ("svm", SVC(C=4.0, kernel="rbf", class_weight="balanced", random_state=seed))]
        ),
        "SVM (Linear, C=1.0, balanced)": Pipeline(
            [("scaler", StandardScaler()), ("svm", SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=seed))]
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced", random_state=seed
        ),
        "Logistic Regression (balanced)": Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", random_state=seed))]
        ),
        "MLP Neural Net (64, 32)": Pipeline(
            [("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed))]
        ),
    }

    rows = []
    for name, model in candidate_models.items():
        preds = np.zeros(len(features), dtype=int)
        scores = np.zeros(len(features), dtype=float)
        for train_idx, test_idx in splitter.split(x_values, y_values, groups):
            model.fit(x_values[train_idx], y_values[train_idx])
            preds[test_idx] = model.predict(x_values[test_idx])
            if hasattr(model, "decision_function"):
                d = model.decision_function(x_values[test_idx])
                scores[test_idx] = d[:, 1] if d.ndim == 2 else d
            else:
                scores[test_idx] = model.predict_proba(x_values[test_idx])[:, 1]

        acc = accuracy_score(y_values, preds)
        rec = recall_score(y_values, preds, pos_label=2, zero_division=0)
        prec = precision_score(y_values, preds, pos_label=2, zero_division=0)
        f1_girls = f1_score(y_values, preds, pos_label=2, zero_division=0)
        auc = roc_auc_score((y_values == 2).astype(int), scores)
        f1_macro = f1_score(y_values, preds, average="macro", zero_division=0)

        rows.append({
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "girls_sensitivity_recall": round(float(rec), 4),
            "girls_precision": round(float(prec), 4),
            "girls_f1": round(float(f1_girls), 4),
            "macro_f1": round(float(f1_macro), 4),
            "roc_auc": round(float(auc), 4),
        })

    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    model: Pipeline,
    benchmarks: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = [column for column in features.columns if column.startswith("mfcc_")]
    features.to_csv(output_dir / "mfcc_features.csv", index=False)
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    if benchmarks is not None:
        benchmarks.to_csv(output_dir / "model_comparison.csv", index=False)
    group_review = (
        features.groupby("group_id", as_index=False)
        .agg(
            classes=("class_name", lambda values: ", ".join(sorted(set(values)))),
            recordings=("filename", "count"),
            class_count=("class_name", "nunique"),
        )
        .sort_values(["class_count", "recordings", "group_id"], ascending=[False, False, True])
    )
    group_review["review_required"] = np.where(
        group_review["class_count"] > 1, "yes - inferred name appears in both labelled folders", "no"
    )
    group_review.to_csv(output_dir / "speaker_group_review.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    joblib.dump({"model": model, "feature_columns": feature_columns}, output_dir / "gender_svm_model.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)

    matrix = np.asarray(metrics["confusion_matrix_rows_actual_columns_predicted"])
    plot_class_distribution(features, output_dir / "01_class_distribution.png")
    plot_pca_scatter(features, feature_columns, output_dir / "02_mfcc_pca_scatter.png")
    plot_direct_mfcc_scatter(features, output_dir / "02b_mfcc_scatter_direct.png")
    plot_confusion_matrix(matrix, output_dir / "03_confusion_matrix.png")
    plot_roc_curve(
        features["label"].to_numpy(dtype=int),
        predictions["girls_decision_score"].to_numpy(dtype=float),
        float(metrics["roc_auc_girls"]),
        output_dir / "04_roc_curve.png",
    )

    included_counts = features["class_name"].value_counts().to_dict()
    excluded_count = int((manifest["status"] == "excluded").sum())
    summary = "\n".join(
        [
            "EC24313 Assessment 2 - Run summary",
            "=" * 38,
            f"Usable recordings: {len(features)} ({included_counts.get('boys', 0)} boys, {included_counts.get('girls', 0)} girls)",
            f"Excluded recordings: {excluded_count}",
            f"Filename-derived speaker groups: {features['group_id'].nunique()}",
            f"Groups needing label review: {(group_review['class_count'] > 1).sum()}",
            f"Evaluation folds: {metrics['folds']}",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Sensitivity/recall (girls): {metrics['sensitivity_recall_girls']:.4f}",
            f"Precision (girls): {metrics['precision_girls']:.4f}",
            f"ROC-AUC (girls): {metrics['roc_auc_girls']:.4f}",
            "",
            "Inspect dataset_manifest.csv and oof_predictions.csv before submission.",
        ]
    )
    (output_dir / "run_summary.txt").write_text(summary + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "outputs"
    dataset_root = prepare_dataset(project_root)
    features, manifest = build_feature_table(dataset_root)
    feature_columns = [column for column in features.columns if column.startswith("mfcc_")]
    if len(feature_columns) != 26:
        raise RuntimeError(f"Expected 26 MFCC summary features, found {len(feature_columns)}")
    metrics, predictions, final_model = evaluate_model(features, feature_columns, args.seed)
    benchmarks = evaluate_model_benchmarks(features, feature_columns, args.seed)
    save_outputs(output_dir, features, manifest, metrics, predictions, final_model, benchmarks)
    print((output_dir / "run_summary.txt").read_text(encoding="utf-8"))
    print("\nModel Benchmark Comparison:")
    print(benchmarks.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
