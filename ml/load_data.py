"""
Train the job-fraud classifier and write ml/model.joblib + ml/vectorizer.joblib.

Primary data: data/fake_job_postings.csv (mixed labels, used for holdout).
Supplemental: data/Fake Postings.csv (fraud-only, train-only, down-weighted).
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CSV = ROOT / "data" / "fake_job_postings.csv"
SUPPLEMENTAL_CSV = ROOT / "data" / "Fake Postings.csv"
MODEL_PATH = ROOT / "ml" / "model.joblib"
VECTORIZER_PATH = ROOT / "ml" / "vectorizer.joblib"

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
SUPPLEMENTAL_SAMPLE_WEIGHT = 0.1


def fuse_text(df: pd.DataFrame) -> pd.Series:
    frame = df.reindex(columns=TEXT_COLS).fillna("")
    return frame.astype(str).agg(" ".join, axis=1).map(clean_text)


def load_labeled_jobs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TEXT_COLS + ["fraudulent"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    out = df[TEXT_COLS + ["fraudulent"]].copy()
    out["fraudulent"] = out["fraudulent"].astype(int)
    out["clean_text"] = fuse_text(out)
    out["source"] = path.name
    return out


def main() -> None:
    primary = load_labeled_jobs(PRIMARY_CSV)
    supplemental = load_labeled_jobs(SUPPLEMENTAL_CSV)

    if not (supplemental["fraudulent"] == 1).all():
        raise ValueError(
            f"{SUPPLEMENTAL_CSV.name} should be exclusively fraudulent; "
            f"got counts:\n{supplemental['fraudulent'].value_counts()}"
        )

    print("Primary shape:", primary.shape)
    print("Primary class balance:\n", primary["fraudulent"].value_counts())
    print("\nSupplemental fraud rows:", len(supplemental))
    print(f"Supplemental sample weight: {SUPPLEMENTAL_SAMPLE_WEIGHT}")

    X_text = primary["clean_text"]
    y = primary["fraudulent"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    n_primary_train = len(y_train)

    X_train_text = pd.concat(
        [X_train_text, supplemental["clean_text"]],
        ignore_index=True,
    )
    y_train = pd.concat(
        [y_train, supplemental["fraudulent"]],
        ignore_index=True,
    )
    sample_weight = np.concatenate(
        [
            np.ones(n_primary_train, dtype=float),
            np.full(len(supplemental), SUPPLEMENTAL_SAMPLE_WEIGHT, dtype=float),
        ]
    )

    print("\nTrain size (with supplemental):", len(y_train))
    print("Train class balance:\n", y_train.value_counts())
    print("Test size (primary holdout only):", len(y_test))
    print("Test class balance:\n", y_test.value_counts())

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\nConfusion matrix @0.5:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:           {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:            {average_precision_score(y_test, y_prob):.4f}")

    print("\nThreshold sweep (fraud class):")
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_hat = (y_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_hat, average="binary", zero_division=0
        )
        print(
            f"Threshold: {threshold:.1f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest F1 threshold on holdout: {best_threshold} (F1={best_f1:.4f})")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
