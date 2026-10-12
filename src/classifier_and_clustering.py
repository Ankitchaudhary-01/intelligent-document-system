"""
Module 3: Supervised Document Classification, Unsupervised K-Means Clustering,
Model Comparison, Bootstrap Confidence Intervals, Slice Analysis, and Visualizations.

Run directly via:
    .venv/bin/python -m src.classifier_and_clustering
"""

import io
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.generate_corpus import generate_corpus
from src.search_engine import DocumentSearchEngine
from src.text_preprocessing import (
    TextColumnSelector,
    audit_and_clean_corpus,
    build_hybrid_preprocessor,
    extract_document_stats,
    preprocess_text,
)
from src.tfidf_engine import TFIDFDocumentEngine


sns.set_theme(style="whitegrid", palette="deep")
RANDOM_SEED = 42


def get_model_size_kb(model: Any) -> float:
    """Measures serialized model size in KB using an in-memory buffer."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return round(len(buf.getvalue()) / 1024.0, 2)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
    n_bootstraps: int = 800,
    seed: int = RANDOM_SEED,
) -> dict[str, list[float]]:
    """Computes 95% Bootstrap Confidence Intervals for Macro F1, Accuracy, and Macro ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_scores, acc_scores, auc_scores = [], [], []
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < n_classes:
            continue
        f1_scores.append(float(f1_score(y_true[idx], y_pred[idx], average="macro")))
        acc_scores.append(float(accuracy_score(y_true[idx], y_pred[idx])))
        auc_scores.append(
            float(roc_auc_score(y_true_bin[idx], y_proba[idx], average="macro", multi_class="ovr"))
        )

    return {
        "macro_f1_ci_95": [round(float(np.percentile(f1_scores, 2.5)), 4), round(float(np.percentile(f1_scores, 97.5)), 4)],
        "accuracy_ci_95": [round(float(np.percentile(acc_scores, 2.5)), 4), round(float(np.percentile(acc_scores, 97.5)), 4)],
        "roc_auc_ci_95": [round(float(np.percentile(auc_scores, 2.5)), 4), round(float(np.percentile(auc_scores, 97.5)), 4)],
    }


def generate_eda_and_tfidf_plots(
    clean_df: pd.DataFrame,
    top_terms_by_cat: dict[str, list[dict[str, Any]]],
    fig_dir: Path,
) -> None:
    """Generates EDA distribution plots and Top TF-IDF terms per category plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Category counts
    cat_counts = clean_df["category"].value_counts()
    sns.barplot(
        x=cat_counts.index,
        y=cat_counts.values,
        hue=cat_counts.index,
        palette="viridis",
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Document Distribution Across Categories", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Document Category")
    axes[0, 0].set_ylabel("Number of Documents")

    # 2. Citation count distribution showing outliers by category
    sns.boxplot(
        data=clean_df,
        x="category",
        y="citation_count",
        hue="category",
        palette="Set2",
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Citation Count Distribution & Outliers by Category", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Document Category")
    axes[0, 1].set_ylabel("Citation Count")

    # 3. Author experience vs Citation count scatter
    sns.scatterplot(
        data=clean_df,
        x="author_experience_years",
        y="citation_count",
        hue="category",
        style="access_tier",
        palette="deep",
        alpha=0.8,
        s=65,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Author Experience vs. Citations by Category & Access Tier", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Author Experience (Years)")
    axes[1, 0].set_ylabel("Citation Count")
    axes[1, 0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # 4. Access Tier composition per Category
    ct = pd.crosstab(clean_df["category"], clean_df["access_tier"], normalize="index") * 100
    ct[["Public", "Internal", "Confidential", "Restricted"]].plot(
        kind="bar",
        stacked=True,
        colormap="Spectral",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Access Tier Proportions by Document Category (%)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Document Category")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].tick_params(axis="x", rotation=0)
    axes[1, 1].legend(title="Access Tier", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "eda_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Top TF-IDF Terms Plot per Category
    categories = sorted(top_terms_by_cat.keys())
    fig, axes = plt.subplots(1, len(categories), figsize=(20, 5.5), sharex=False)
    palette = sns.color_palette("husl", len(categories))

    for idx, cat in enumerate(categories):
        terms_df = pd.DataFrame(top_terms_by_cat[cat][:8])
        sns.barplot(
            data=terms_df,
            x="mean_tfidf",
            y="term",
            color=palette[idx],
            ax=axes[idx],
        )
        axes[idx].set_title(f"{cat}\nTop TF-IDF Terms", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Mean TF-IDF Weight")
        axes[idx].set_ylabel("")

    plt.tight_layout()
    plt.savefig(fig_dir / "tfidf_top_terms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_classification_plots(
    val_results: dict[str, dict[str, Any]],
    val_probas: dict[str, np.ndarray],
    y_val: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    class_names: list[str],
    selected_model_name: str,
    fig_dir: Path,
) -> None:
    """Generates model comparison bar chart, ROC curves, calibration curves, and confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    # 1. Model Comparison Bar Chart (Train F1 vs CV F1 vs Val F1)
    model_names = list(val_results.keys())
    short_names = [m.replace(" (Regularized)", " (Reg)").replace(" (Unregularized)", " (Unreg)") for m in model_names]
    x_pos = np.arange(len(model_names))
    width = 0.26

    train_f1s = [val_results[m]["train_macro_f1"] for m in model_names]
    cv_f1s = [val_results[m]["cv_macro_f1_mean"] for m in model_names]
    val_f1s = [val_results[m]["val_macro_f1"] for m in model_names]

    axes[0].bar(x_pos - width, train_f1s, width, label="Train Macro F1", color="#4C72B0")
    axes[0].bar(x_pos, cv_f1s, width, label="5-Fold CV Macro F1", color="#55A868")
    axes[0].bar(x_pos + width, val_f1s, width, label="Validation Macro F1", color="#C44E52")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(short_names, rotation=28, ha="right", fontsize=9)
    axes[0].set_ylim(0.0, 1.1)
    axes[0].set_ylabel("Macro F1-Score")
    axes[0].set_title("Model Comparison: Train vs. CV vs. Validation F1", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)

    # 2. Macro-Average ROC Curves on Validation Set
    y_val_bin = label_binarize(y_val, classes=list(range(len(class_names))))
    for m_name, proba in val_probas.items():
        fpr, tpr, _ = roc_curve(y_val_bin.ravel(), proba.ravel())
        auc_val = val_results[m_name]["val_roc_auc"]
        axes[1].plot(fpr, tpr, lw=2, label=f"{m_name[:22]} (AUC={auc_val:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Micro/Macro One-vs-Rest ROC Curves (Validation)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8)

    # 3. Calibration Curve (Reliability Diagram) on Top Models
    for m_name in [
        "Logistic Regression (L2)",
        "Calibrated Linear SVM (L2)",
        "Random Forest (Regularized)",
        "Decision Tree (Regularized)",
    ]:
        if m_name in val_probas:
            proba = val_probas[m_name]
            prob_true, prob_pred = calibration_curve(y_val_bin.ravel(), proba.ravel(), n_bins=8)
            axes[2].plot(prob_pred, prob_true, marker="o", lw=2, label=m_name[:24])
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly Calibrated")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction of Positives")
    axes[2].set_title("Probability Calibration Curves", fontsize=12, fontweight="bold")
    axes[2].legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(fig_dir / "classification_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Confusion Matrices: Unregularized Tree vs Regularized Tree vs Selected Model on Test Set
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    cm_unreg = np.array(val_results["Decision Tree (Unregularized)"]["confusion_matrix"])
    cm_reg = np.array(val_results["Decision Tree (Regularized)"]["confusion_matrix"])
    cm_test = confusion_matrix(y_test, y_test_pred)

    for ax, cm, title, cmap in [
        (axes[0], cm_unreg, "Val: Decision Tree (Unregularized)", "Oranges"),
        (axes[1], cm_reg, "Val: Decision Tree (Regularized)", "Purples"),
        (axes[2], cm_test, f"Held-Out Test: {selected_model_name}", "Blues"),
    ]:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Category")
        ax.set_ylabel("True Category")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_clustering_analysis(
    clean_df: pd.DataFrame,
    tfidf_matrix: np.ndarray,
    tfidf_engine: TFIDFDocumentEngine,
    fig_dir: Path,
) -> tuple[np.ndarray, PCA, TruncatedSVD, StandardScaler, KMeans, dict[str, Any]]:
    """
    Executes Unsupervised K-Means Clustering following ML Best Practices:
    1. Standardizes combined lexical (SVD-reduced TF-IDF) + numerical/encoded metadata features.
    2. Sweeps k = 2..8 and collects Silhouette Score & Inertia.
    3. Selects optimal k* maximizing Silhouette Score.
    4. Reduces feature space to 2D using PCA (`PCA(n_components=2)`).
    5. Plots 2D PCA scatterplots colored by Cluster Labels and by Discrete Category/Access Tier.
    6. Summarizes each cluster by feature distributions and top TF-IDF terms.
    """
    # Combine top 5 SVD semantic components of TF-IDF with scaled metadata features
    svd = TruncatedSVD(n_components=5, random_state=RANDOM_SEED)
    lexical_components = svd.fit_transform(tfidf_matrix)

    ord_map = {"Public": 0.0, "Internal": 1.0, "Confidential": 2.0, "Restricted": 3.0}
    meta_df = pd.DataFrame(
        {
            "exp_years": clean_df["author_experience_years"].fillna(clean_df["author_experience_years"].median()),
            "citations": clean_df["citation_count"].fillna(clean_df["citation_count"].median()),
            "access_ord": clean_df["access_tier"].map(ord_map).fillna(1.0),
            "is_verified": clean_df["is_verified"].astype(float),
        }
    )
    scaler = StandardScaler()
    scaled_meta = scaler.fit_transform(meta_df)

    # Weight lexical features prominently so clusters reflect document topics & metadata
    clustering_features = np.hstack([lexical_components * 4.0, scaled_meta * 0.15])
    clustering_features = Normalizer(norm="l2").fit_transform(clustering_features)

    k_range = list(range(2, 9))
    silhouette_scores: dict[int, float] = {}
    inertias: dict[int, float] = {}
    models_by_k: dict[int, KMeans] = {}

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
        labels = km.fit_predict(clustering_features)
        sil = float(silhouette_score(clustering_features, labels))
        silhouette_scores[k] = round(sil, 4)
        inertias[k] = round(float(km.inertia_), 2)
        models_by_k[k] = km

    optimal_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
    best_kmeans = models_by_k[optimal_k]
    cluster_labels = best_kmeans.labels_

    # Project into 2 dimensions using PCA for visualization
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    coords_2d = pca_2d.fit_transform(clustering_features)

    clean_df["cluster_id"] = cluster_labels
    clean_df["pca_1"] = coords_2d[:, 0]
    clean_df["pca_2"] = coords_2d[:, 1]

    # Plot Silhouette Curve + PCA 2D by Cluster + PCA 2D by Discrete Category
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    axes[0].plot(
        list(silhouette_scores.keys()),
        list(silhouette_scores.values()),
        marker="o",
        lw=2.5,
        color="#2B5B84",
    )
    axes[0].axvline(optimal_k, color="#C44E52", linestyle="--", lw=1.8, label=f"Optimal k = {optimal_k}")
    axes[0].set_title("Silhouette Score Across Number of Clusters (k)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_xticks(k_range)
    axes[0].legend()

    sns.scatterplot(
        data=clean_df,
        x="pca_1",
        y="pca_2",
        hue=clean_df["cluster_id"].apply(lambda c: f"Cluster {c}"),
        palette="tab10",
        s=60,
        alpha=0.85,
        ax=axes[1],
    )
    axes[1].set_title(f"2D PCA Projection Colored by Cluster Label (k={optimal_k})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"PCA Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PCA Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[1].legend(title="Cluster ID", loc="best", fontsize=8.5)

    sns.scatterplot(
        data=clean_df,
        x="pca_1",
        y="pca_2",
        hue="category",
        style="access_tier",
        palette="Set1",
        s=60,
        alpha=0.85,
        ax=axes[2],
    )
    axes[2].set_title("2D PCA Projection Colored by Discrete Category Feature", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(f"PCA Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[2].set_ylabel(f"PCA Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "clustering_silhouette_and_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Describe each cluster by feature distributions and top TF-IDF terms
    cluster_top_terms = tfidf_engine.get_top_terms_by_group(tfidf_matrix, cluster_labels, top_n=8)
    cluster_descriptions = {}
    for c_id in sorted(np.unique(cluster_labels)):
        sub = clean_df[clean_df["cluster_id"] == c_id]
        dom_cat = str(sub["category"].mode().iloc[0])
        cat_purity = round(float((sub["category"] == dom_cat).mean() * 100), 2)
        dom_access = str(sub["access_tier"].mode().iloc[0])
        dom_channel = str(sub["source_channel"].mode().iloc[0])
        top_words = [t["term"] for t in cluster_top_terms.get(str(c_id), [])[:6]]

        cluster_descriptions[f"Cluster {c_id}"] = {
            "cluster_id": int(c_id),
            "document_count": int(len(sub)),
            "dominant_category": dom_cat,
            "dominant_category_purity_pct": cat_purity,
            "top_tfidf_terms": top_words,
            "mean_author_experience_years": round(float(sub["author_experience_years"].mean()), 2),
            "mean_citation_count": round(float(sub["citation_count"].mean()), 2),
            "typical_access_tier": dom_access,
            "typical_source_channel": dom_channel,
        }

    clustering_report = {
        "k_range_evaluated": k_range,
        "silhouette_scores_by_k": {str(k): v for k, v in silhouette_scores.items()},
        "inertias_by_k": {str(k): v for k, v in inertias.items()},
        "optimal_k": int(optimal_k),
        "optimal_silhouette_score": silhouette_scores[optimal_k],
        "pca_explained_variance_ratio": [
            round(float(v), 4) for v in pca_2d.explained_variance_ratio_
        ],
        "cluster_profiles": cluster_descriptions,
    }

    return cluster_labels, pca_2d, svd, scaler, best_kmeans, clustering_report


def train_and_evaluate_all() -> dict[str, Any]:
    """Runs the end-to-end data generation, preprocessing, TF-IDF, classification, clustering, and search indexing."""
    corpus_path = Path("data/documents_corpus.csv")
    if not corpus_path.exists():
        raw_df = generate_corpus(corpus_path)
    else:
        raw_df = pd.read_csv(corpus_path)

    # 1. Audit and clean corpus
    clean_df, audit_report = audit_and_clean_corpus(raw_df)

    # 2. Strict Featurization Ordering: Split BEFORE fitting any transformers (70% Train / 15% Val / 15% Test)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(clean_df["category"])
    class_names = list(label_encoder.classes_)

    X_train, X_temp, y_train, y_temp = train_test_split(
        clean_df,
        y_all,
        test_size=0.30,
        stratify=y_all,
        random_state=RANDOM_SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    # 3. Fit TFIDFDocumentEngine strictly on X_train["clean_text"], then index clean_df
    tfidf_engine = TFIDFDocumentEngine(max_features=800, ngram_range=(1, 2))
    tfidf_engine.fit(X_train["clean_text"])
    corpus_tfidf_matrix = tfidf_engine.index_corpus(clean_df["clean_text"])
    top_terms_by_category = tfidf_engine.get_top_terms_by_group(
        corpus_tfidf_matrix, clean_df["category"], top_n=10
    )

    # 4. Build and evaluate candidate classification pipelines on identical splits
    candidate_estimators = {
        "Naive Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
        "Complement Naive Bayes (TF-IDF)": Pipeline(
            steps=[
                ("selector", TextColumnSelector("clean_text")),
                ("tfidf", tfidf_engine.vectorizer),
                ("clf", ComplementNB(alpha=0.5)),
            ]
        ),
        "Decision Tree (Unregularized)": Pipeline(
            steps=[
                ("prep", build_hybrid_preprocessor()),
                ("clf", DecisionTreeClassifier(max_depth=None, random_state=RANDOM_SEED)),
            ]
        ),
        "Decision Tree (Regularized)": Pipeline(
            steps=[
                ("prep", build_hybrid_preprocessor()),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_leaf=4,
                        ccp_alpha=0.004,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest (Regularized)": Pipeline(
            steps=[
                ("prep", build_hybrid_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Calibrated Linear SVM (L2)": Pipeline(
            steps=[
                ("prep", build_hybrid_preprocessor()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(C=0.5, random_state=RANDOM_SEED),
                        cv=3,
                    ),
                ),
            ]
        ),
        "Logistic Regression (L2)": Pipeline(
            steps=[
                ("prep", build_hybrid_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    val_results: dict[str, dict[str, Any]] = {}
    val_probas: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    y_val_bin = label_binarize(y_val, classes=list(range(len(class_names))))

    for name, model in candidate_estimators.items():
        if name == "Complement Naive Bayes (TF-IDF)":
            # Clone a dedicated pipeline so vectorizer is fit inside CV without leakage
            from sklearn.feature_extraction.text import TfidfVectorizer
            model = Pipeline(
                steps=[
                    ("selector", TextColumnSelector("clean_text")),
                    ("tfidf", TfidfVectorizer(max_features=800, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                    ("clf", ComplementNB(alpha=0.5)),
                ]
            )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro")

        y_train_pred = model.predict(X_train)
        t_inf0 = time.perf_counter()
        y_val_pred = model.predict(X_val)
        inf_latency_ms = ((time.perf_counter() - t_inf0) * 1000.0) / len(X_val)
        y_val_proba = model.predict_proba(X_val)

        train_f1 = float(f1_score(y_train, y_train_pred, average="macro"))
        val_acc = float(accuracy_score(y_val, y_val_pred))
        val_prec = float(precision_score(y_val, y_val_pred, average="macro", zero_division=0))
        val_rec = float(recall_score(y_val, y_val_pred, average="macro", zero_division=0))
        val_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        val_roc_auc = float(roc_auc_score(y_val_bin, y_val_proba, average="macro", multi_class="ovr"))
        val_pr_auc = float(average_precision_score(y_val_bin, y_val_proba, average="macro"))

        val_results[name] = {
            "train_macro_f1": round(train_f1, 4),
            "cv_macro_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_macro_f1_std": round(float(cv_scores.std()), 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_precision": round(val_prec, 4),
            "val_macro_recall": round(val_rec, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_roc_auc": round(val_roc_auc, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "train_time_ms": round(train_time_ms, 2),
            "inference_latency_ms_per_doc": round(inf_latency_ms, 4),
            "model_size_kb": get_model_size_kb(model),
            "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist(),
        }
        val_probas[name] = y_val_proba
        fitted_models[name] = model

    # 5. Select optimal production classifier (Logistic Regression L2) and evaluate on Held-Out Test Set
    selected_model_name = "Logistic Regression (L2)"
    best_clf = fitted_models[selected_model_name]

    y_test_pred = best_clf.predict(X_test)
    y_test_proba = best_clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

    test_ci = bootstrap_ci(y_test, y_test_pred, y_test_proba, n_classes=len(class_names))

    # Slice-based error analysis on Test Set across subpopulations (`access_tier` and `source_channel`)
    slice_analysis: dict[str, dict[str, Any]] = {"by_access_tier": {}, "by_source_channel": {}}
    X_test_reset = X_test.reset_index(drop=True)
    for tier in sorted(X_test_reset["access_tier"].unique()):
        mask = (X_test_reset["access_tier"] == tier).to_numpy()
        if mask.sum() > 0:
            slice_analysis["by_access_tier"][str(tier)] = {
                "sample_count": int(mask.sum()),
                "accuracy": round(float(accuracy_score(y_test[mask], y_test_pred[mask])), 4),
                "macro_f1": round(float(f1_score(y_test[mask], y_test_pred[mask], average="macro", zero_division=0)), 4),
            }
    for channel in sorted(X_test_reset["source_channel"].unique()):
        mask = (X_test_reset["source_channel"] == channel).to_numpy()
        if mask.sum() > 0:
            slice_analysis["by_source_channel"][str(channel)] = {
                "sample_count": int(mask.sum()),
                "accuracy": round(float(accuracy_score(y_test[mask], y_test_pred[mask])), 4),
                "macro_f1": round(float(f1_score(y_test[mask], y_test_pred[mask], average="macro", zero_division=0)), 4),
            }

    # Extract top logistic regression feature weights per category for model interpretability
    prep_step: ColumnTransformer = best_clf.named_steps["prep"]
    lr_step: LogisticRegression = best_clf.named_steps["clf"]
    feature_names_out = prep_step.get_feature_names_out()
    top_weights_per_class: dict[str, list[dict[str, float | str]]] = {}
    for cls_idx, cls_name in enumerate(class_names):
        coefs = lr_step.coef_[cls_idx]
        top_idx = np.argsort(coefs)[::-1][:10]
        top_weights_per_class[cls_name] = [
            {
                "feature": str(feature_names_out[i]).replace("tfidf__", "").replace("num__", "").replace("nom__", ""),
                "weight": round(float(coefs[i]), 4),
            }
            for i in top_idx
        ]

    # 6. Run Unsupervised K-Means Clustering & Generate All Diagnostic Figures
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    generate_eda_and_tfidf_plots(clean_df, top_terms_by_category, fig_dir)
    generate_classification_plots(
        val_results,
        val_probas,
        y_val,
        y_test,
        y_test_pred,
        class_names,
        selected_model_name,
        fig_dir,
    )
    cluster_labels, pca_2d, svd, cluster_scaler, best_kmeans, clustering_report = run_clustering_analysis(
        clean_df,
        corpus_tfidf_matrix,
        tfidf_engine,
        fig_dir,
    )

    # 7. Build Search Engine over the indexed corpus
    search_engine = DocumentSearchEngine(
        corpus_df=clean_df,
        tfidf_engine=tfidf_engine,
        cluster_labels=cluster_labels,
    )

    # 8. Compile and save metrics report + serialized model bundle
    final_test_evaluation = {
        "selected_model": selected_model_name,
        "test_sample_count": int(len(y_test)),
        "test_accuracy": round(float(accuracy_score(y_test, y_test_pred)), 4),
        "test_macro_precision": round(float(precision_score(y_test, y_test_pred, average="macro")), 4),
        "test_macro_recall": round(float(recall_score(y_test, y_test_pred, average="macro")), 4),
        "test_macro_f1": round(float(f1_score(y_test, y_test_pred, average="macro")), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test_bin, y_test_proba, average="macro", multi_class="ovr")), 4),
        "test_pr_auc": round(float(average_precision_score(y_test_bin, y_test_proba, average="macro")), 4),
        "bootstrap_95_confidence_intervals": test_ci,
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "class_names": class_names,
        "slice_based_error_analysis": slice_analysis,
        "interpretability_top_features_per_class": top_weights_per_class,
    }

    summary_report = {
        "data_hygiene_and_schema_audit": audit_report,
        "split_sizes": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "tfidf_vocabulary_size": int(len(tfidf_engine.feature_names)),
        "top_tfidf_terms_by_category": top_terms_by_category,
        "validation_model_comparison": val_results,
        "final_test_evaluation": final_test_evaluation,
        "clustering_evaluation": clustering_report,
    }

    report_path = Path("reports/metrics_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")

    model_bundle = {
        "classifier_pipeline": best_clf,
        "label_encoder": label_encoder,
        "tfidf_engine": tfidf_engine,
        "kmeans_model": best_kmeans,
        "svd_model": svd,
        "cluster_scaler": cluster_scaler,
        "pca_2d": pca_2d,
        "search_engine": search_engine,
        "cluster_profiles": clustering_report["cluster_profiles"],
    }
    model_path = Path("models/document_intelligence.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"Saved metrics report to '{report_path}' and model bundle to '{model_path}'.")
    print(
        f"Selected Classifier ({selected_model_name}) Test Macro F1: "
        f"{final_test_evaluation['test_macro_f1']:.4f} | Test ROC-AUC: {final_test_evaluation['test_roc_auc']:.4f}"
    )
    print(
        f"Optimal K-Means Clusters: k = {clustering_report['optimal_k']} "
        f"(Silhouette Score = {clustering_report['optimal_silhouette_score']:.4f})"
    )
    return summary_report


def analyze_single_document(
    title: str,
    content: str,
    access_tier: str = "Internal",
    source_channel: str = "Enterprise_Wiki",
    author_experience_years: float = 8.0,
    citation_count: float = 20.0,
    is_verified: int = 1,
    published_date: str = "2025-05-01",
    bundle_path: str | Path = "models/document_intelligence.joblib",
) -> dict[str, Any]:
    """
    Runs end-to-end NLP preprocessing, TF-IDF term weighting, Supervised Category Classification,
    Unsupervised Cluster Assignment, and Cosine Similarity Retrieval on any new document.
    """
    bundle = joblib.load(bundle_path)
    clf = bundle["classifier_pipeline"]
    le: LabelEncoder = bundle["label_encoder"]
    tfidf_engine: TFIDFDocumentEngine = bundle["tfidf_engine"]
    kmeans: KMeans = bundle["kmeans_model"]
    svd: TruncatedSVD = bundle["svd_model"]
    cluster_scaler: StandardScaler = bundle["cluster_scaler"]
    search_engine: DocumentSearchEngine = bundle["search_engine"]

    raw_full = f"{title}. {content}"
    nlp_stats = extract_document_stats(raw_full)
    tfidf_breakdown = tfidf_engine.inspect_document_tfidf(raw_full, top_n=12)

    pub_ts = float((pd.Timestamp(published_date) - pd.Timestamp("1970-01-01")).total_seconds())
    input_df = pd.DataFrame(
        [
            {
                "clean_text": nlp_stats["preprocessed_text"],
                "published_timestamp": pub_ts,
                "author_experience_years": float(author_experience_years),
                "citation_count": float(citation_count),
                "word_count": float(nlp_stats["raw_word_count"]),
                "access_tier": access_tier,
                "source_channel": source_channel,
                "is_verified": int(is_verified),
            }
        ]
    )

    pred_idx = int(clf.predict(input_df)[0])
    pred_probas = clf.predict_proba(input_df)[0]
    predicted_category = str(le.inverse_transform([pred_idx])[0])
    confidence_pct = round(float(pred_probas[pred_idx] * 100), 2)

    class_probabilities = {
        str(cls_name): round(float(prob * 100), 2)
        for cls_name, prob in zip(le.classes_, pred_probas)
    }

    # Predict cluster assignment
    doc_tfidf_vec = tfidf_engine.transform_text(nlp_stats["preprocessed_text"], is_preprocessed=True).reshape(1, -1)
    lex_comp = svd.transform(doc_tfidf_vec)
    ord_map = {"Public": 0.0, "Internal": 1.0, "Confidential": 2.0, "Restricted": 3.0}
    meta_arr = pd.DataFrame(
        [
            {
                "exp_years": float(author_experience_years),
                "citations": float(citation_count),
                "access_ord": float(ord_map.get(access_tier, 1.0)),
                "is_verified": float(is_verified),
            }
        ]
    )
    scaled_meta = cluster_scaler.transform(meta_arr)
    clust_feat = Normalizer(norm="l2").fit_transform(np.hstack([lex_comp * 4.0, scaled_meta * 0.15]))
    assigned_cluster = int(kmeans.predict(clust_feat)[0])
    cluster_profile = bundle["cluster_profiles"].get(f"Cluster {assigned_cluster}", {})

    similar_docs = search_engine.similarity_search(raw_full, top_k=5)

    return {
        "predicted_category": predicted_category,
        "confidence_percentage": f"{confidence_pct:.2f}%",
        "class_probabilities": class_probabilities,
        "assigned_cluster_id": assigned_cluster,
        "assigned_cluster_profile": cluster_profile,
        "nlp_preprocessing_stats": nlp_stats,
        "tfidf_term_table": tfidf_breakdown.to_dict(orient="records"),
        "top_similar_documents": similar_docs,
    }


if __name__ == "__main__":
    train_and_evaluate_all()
