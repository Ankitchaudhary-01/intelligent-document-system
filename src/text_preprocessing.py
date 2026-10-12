"""
Module 1: Document Preprocessing, Schema Audit, and Leak-Free Feature Engineering.

Responsibilities:
1. NLP Text Preprocessing:
   - Lowercasing, special character/punctuation removal, whitespace normalization
   - Tokenization and English stopword removal
   - Rule-based morphological lemmatization (preserving readable root words)
2. Schema Audit & Data Hygiene:
   - Identifies and logs rows that violate domain/schema constraints
   - Identifies and logs rows missing the target label (`category`)
   - Preserves rows with missing metadata (`author_experience_years`, `citation_count`)
     and creates missingness indicators + median imputation inside the training pipeline
3. Leak-Free Scikit-Learn ColumnTransformer:
   - Transforms `published_date` -> numerical `published_timestamp`
   - Preserves pre-encoded binary `is_verified` (`0`/`1`)
   - Encodes ordinal `access_tier` with `OrdinalEncoder`
   - Encodes nominal `source_channel` with `OneHotEncoder`
   - Standardizes numerical features with `StandardScaler`
"""

import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "document",
        "study",
        "division",
        "investigates",
        "focusing",
        "methodology",
        "integrates",
        "alongside",
        "address",
        "operational",
        "challenges",
        "comprehensive",
        "report",
        "evaluate",
        "systematic",
        "methodologies",
        "empirical",
        "benchmarks",
        "multiple",
        "environments",
        "furthermore",
        "statistical",
        "analysis",
        "confirms",
        "significant",
        "improvements",
        "key",
        "stakeholders",
        "review",
        "findings",
        "align",
        "strategic",
        "planning",
        "long",
        "term",
        "institutional",
        "objectives",
        "experimental",
        "results",
        "longitudinal",
        "observations",
        "validate",
        "theoretical",
        "framework",
        "established",
        "prior",
        "literature",
        "additional",
        "sensitivity",
        "analyses",
        "highlight",
        "importance",
        "continuous",
        "monitoring",
        "quality",
        "assurance",
        "protocols",
        "comparative",
        "evaluation",
        "considers",
        "cross",
        "functional",
        "aspects",
    }
)

# Readable morphological normalization map for domain plurals/inflections
LEMMA_SUFFIX_RULES = [
    ("ies", "y"),
    ("clusters", "cluster"),
    ("networks", "network"),
    ("models", "model"),
    ("systems", "system"),
    ("markets", "market"),
    ("trials", "trial"),
    ("patients", "patient"),
    ("damages", "damage"),
    ("claims", "claim"),
    ("swaps", "swap"),
    ("curves", "curve"),
    ("derivatives", "derivative"),
    ("biomarkers", "biomarker"),
    ("ecosystems", "ecosystem"),
    ("protocols", "protocol"),
    ("weights", "weight"),
]


def lemmatize_token(token: str) -> str:
    """Applies lightweight, human-readable morphological normalization to a token."""
    if len(token) <= 3:
        return token
    for suffix, replacement in LEMMA_SUFFIX_RULES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)] + replacement
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    return token


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Full NLP preprocessing pipeline for a raw document string:
    1. Lowercase conversion
    2. Strip punctuation, digits, and non-alphabetic characters
    3. Tokenization
    4. Stopword removal (English + domain boilerplate stopwords)
    5. Morphological lemmatization
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z\s]", " ", lowered)
    tokens = cleaned.split()
    processed_tokens = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if remove_stopwords and tok in CUSTOM_STOP_WORDS:
            continue
        lemma = lemmatize_token(tok)
        if remove_stopwords and lemma in CUSTOM_STOP_WORDS:
            continue
        processed_tokens.append(lemma)
    return " ".join(processed_tokens)


def extract_document_stats(text: str) -> dict[str, Any]:
    """Computes lexical statistics for a single document (raw vs preprocessed)."""
    raw_words = text.split() if isinstance(text, str) else []
    clean_str = preprocess_text(text)
    clean_words = clean_str.split() if clean_str else []
    unique_terms = sorted(set(clean_words))
    avg_word_len = float(np.mean([len(w) for w in clean_words])) if clean_words else 0.0
    return {
        "raw_word_count": len(raw_words),
        "clean_token_count": len(clean_words),
        "unique_token_count": len(unique_terms),
        "lexical_diversity": round(len(unique_terms) / max(len(clean_words), 1), 4),
        "avg_token_length": round(avg_word_len, 2),
        "preprocessed_text": clean_str,
        "sample_tokens": clean_words[:20],
    }


def audit_and_clean_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Audits the raw document corpus for:
    1. Rows that don't make sense (schema/domain anomalies)
    2. Rows without a target value (`category` is NaN)
    3. Missing value frequencies across all columns
    4. Date-to-timestamp transformation and combined NLP text field creation
    """
    total_initial_rows = len(df)

    # 1. Identify rows that don't make sense (anomalies)
    empty_content_mask = df["content"].isna() | (df["content"].astype(str).str.strip() == "")
    invalid_exp_mask = df["author_experience_years"].notna() & (
        (df["author_experience_years"] < 0) | (df["author_experience_years"] > 80)
    )
    invalid_cit_mask = df["citation_count"].notna() & (df["citation_count"] < 0)
    invalid_ver_mask = ~df["is_verified"].isin([0, 1])

    anomaly_mask = empty_content_mask | invalid_exp_mask | invalid_cit_mask | invalid_ver_mask
    anomalous_rows_df = df[anomaly_mask].copy()

    # 2. Identify rows without a target value
    missing_target_mask = df["category"].isna() | (df["category"].astype(str).str.strip() == "")
    missing_target_df = df[missing_target_mask & ~anomaly_mask].copy()

    # 3. Drop anomalous rows and unlabeled target rows for supervised/clustering corpus
    clean_df = df[~anomaly_mask & ~missing_target_mask].copy().reset_index(drop=True)

    # 4. Transform dates into numerical timestamps (seconds since epoch)
    clean_df["published_timestamp"] = (
        pd.to_datetime(clean_df["published_date"], errors="coerce") - pd.Timestamp("1970-01-01")
    ).dt.total_seconds().astype(float)

    # 5. Create combined raw text and preprocessed NLP text columns
    clean_df["raw_full_text"] = clean_df["title"].astype(str) + ". " + clean_df["content"].astype(str)
    clean_df["clean_text"] = clean_df["raw_full_text"].apply(preprocess_text)
    clean_df["word_count"] = clean_df["raw_full_text"].apply(lambda s: len(s.split()))

    missing_summary = {
        col: {
            "missing_count": int(clean_df[col].isna().sum()),
            "missing_pct": round(float(clean_df[col].isna().mean() * 100), 2),
        }
        for col in clean_df.columns
    }

    audit_report = {
        "initial_row_count": int(total_initial_rows),
        "anomalous_row_count": int(len(anomalous_rows_df)),
        "anomalous_rows": anomalous_rows_df[
            ["doc_id", "title", "author_experience_years", "citation_count", "is_verified", "category"]
        ].to_dict(orient="records"),
        "missing_target_row_count": int(len(missing_target_df)),
        "missing_target_rows": missing_target_df[
            ["doc_id", "title", "access_tier", "source_channel", "category"]
        ].to_dict(orient="records"),
        "clean_row_count": int(len(clean_df)),
        "missing_value_audit_on_clean_corpus": missing_summary,
    }

    return clean_df, audit_report


class TextColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single text column as a 1D iterable of strings for TfidfVectorizer."""

    def __init__(self, column: str = "clean_text"):
        self.column = column

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TextColumnSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        return X[self.column].fillna("").astype(str)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array([self.column], dtype=object)


def build_hybrid_preprocessor(max_tfidf_features: int = 600, ngram_range: tuple[int, int] = (1, 2)) -> ColumnTransformer:
    """
    Constructs a leak-free Scikit-Learn ColumnTransformer that combines:
    - TF-IDF vectorization on `clean_text`
    - Numerical pipeline (`SimpleImputer(strategy='median', add_indicator=True)` + `StandardScaler`)
      on `published_timestamp`, `author_experience_years`, `citation_count`, `word_count`
    - Pre-encoded binary feature `is_verified` (mode imputation only, preserving 0/1 encoding)
    - Ordinal pipeline (`OrdinalEncoder` + `StandardScaler`) on `access_tier`
    - Nominal pipeline (`OneHotEncoder`) on `source_channel`
    """
    numerical_cols = ["published_timestamp", "author_experience_years", "citation_count"]
    ordinal_cols = ["access_tier"]
    nominal_cols = ["source_channel"]
    pre_encoded_cols = ["is_verified"]

    tfidf_pipe = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    ord_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["Public", "Internal", "Confidential", "Restricted"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    nom_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    bin_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf_pipe, "clean_text"),
            ("num", num_pipe, numerical_cols),
            ("ord", ord_pipe, ordinal_cols),
            ("nom", nom_pipe, nominal_cols),
            ("bin", bin_pipe, pre_encoded_cols),
        ],
        transformer_weights={
            "tfidf": 3.5,
            "num": 0.35,
            "ord": 0.35,
            "nom": 0.35,
            "bin": 0.25,
        },
        sparse_threshold=0.0,  # Output dense matrix for compatibility across all classifiers & PCA/clustering
    )
    return preprocessor
