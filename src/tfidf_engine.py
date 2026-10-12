"""
Module 2: TF-IDF Feature Extraction & Lexical Analytics Engine.

Responsibilities:
1. Fits a dedicated TfidfVectorizer on the training corpus (strictly leak-free)
   and transforms documents into L2-normalized TF-IDF vectors.
2. Breaks down TF (Term Frequency), IDF (Inverse Document Frequency), and
   combined TF-IDF weights for any single document or user-submitted text.
3. Extracts the top discriminative unigrams and bigrams per category or cluster.
4. Builds an Inverted Index mapping vocabulary terms -> posting lists of (doc_idx, tfidf_weight)
   for fast keyword retrieval.
"""

from collections import Counter, defaultdict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.text_preprocessing import preprocess_text


class TFIDFDocumentEngine:
    """Encapsulates TF-IDF vectorization, term-weight inspection, and inverted index construction."""

    def __init__(self, max_features: int = 800, ngram_range: tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            norm="l2",
        )
        self.feature_names: np.ndarray = np.array([])
        self.idf_map: dict[str, float] = {}
        self.corpus_tfidf_matrix: np.ndarray | None = None
        self.inverted_index: dict[str, list[tuple[int, float]]] = {}

    def fit(self, train_clean_texts: pd.Series | list[str]) -> "TFIDFDocumentEngine":
        """Fits the TF-IDF vocabulary and IDF weights strictly on the training corpus."""
        self.vectorizer.fit(train_clean_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.idf_map = dict(zip(self.feature_names, self.vectorizer.idf_))
        return self

    def index_corpus(self, all_clean_texts: pd.Series | list[str]) -> np.ndarray:
        """Transforms the full clean corpus using the fitted vectorizer and builds an inverted index."""
        sparse_mat = self.vectorizer.transform(all_clean_texts)
        self.corpus_tfidf_matrix = sparse_mat.toarray()

        inv_idx: dict[str, list[tuple[int, float]]] = defaultdict(list)
        coo = sparse_mat.tocoo()
        for doc_idx, term_idx, weight in zip(coo.row, coo.col, coo.data):
            term = str(self.feature_names[term_idx])
            inv_idx[term].append((int(doc_idx), float(weight)))

        # Sort each posting list by descending TF-IDF weight
        self.inverted_index = {
            term: sorted(postings, key=lambda x: x[1], reverse=True)
            for term, postings in inv_idx.items()
        }
        return self.corpus_tfidf_matrix

    def transform_text(self, raw_or_clean_text: str, is_preprocessed: bool = False) -> np.ndarray:
        """Transforms a single document string into a 1D TF-IDF vector."""
        clean_str = raw_or_clean_text if is_preprocessed else preprocess_text(raw_or_clean_text)
        return self.vectorizer.transform([clean_str]).toarray()[0]

    def inspect_document_tfidf(self, raw_text: str, top_n: int = 15) -> pd.DataFrame:
        """
        Returns a detailed breakdown of Term Frequency (TF), IDF score, and
        L2-normalized TF-IDF weight for the top terms in a single document.
        """
        clean_str = preprocess_text(raw_text)
        tokens = clean_str.split()
        unigram_counts = Counter(tokens)
        bigram_counts = Counter(
            f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)
        )
        all_counts = {**unigram_counts, **bigram_counts}

        vec = self.vectorizer.transform([clean_str]).toarray()[0]
        non_zero_indices = np.where(vec > 0)[0]

        rows = []
        for idx in non_zero_indices:
            term = str(self.feature_names[idx])
            raw_tf = int(all_counts.get(term, 1))
            idf_val = float(self.idf_map.get(term, 0.0))
            tfidf_weight = float(vec[idx])
            rows.append(
                {
                    "term": term,
                    "ngram_type": "Bigram (2-word)" if " " in term else "Unigram (1-word)",
                    "raw_term_frequency": raw_tf,
                    "idf_score": round(idf_val, 4),
                    "tfidf_weight": round(tfidf_weight, 4),
                }
            )

        df_terms = pd.DataFrame(rows)
        if df_terms.empty:
            return pd.DataFrame(
                columns=["term", "ngram_type", "raw_term_frequency", "idf_score", "tfidf_weight"]
            )
        return df_terms.sort_values(by="tfidf_weight", ascending=False).head(top_n).reset_index(drop=True)

    def get_top_terms_by_group(
        self, tfidf_matrix: np.ndarray, group_labels: pd.Series | list[Any], top_n: int = 10
    ) -> dict[str, list[dict[str, float | str]]]:
        """Computes the top discriminative TF-IDF terms for each category or cluster label."""
        labels_arr = np.asarray(group_labels)
        unique_groups = sorted(set(labels_arr))
        result: dict[str, list[dict[str, float | str]]] = {}

        for grp in unique_groups:
            mask = labels_arr == grp
            mean_vec = tfidf_matrix[mask].mean(axis=0)
            top_indices = np.argsort(mean_vec)[::-1][:top_n]
            result[str(grp)] = [
                {
                    "term": str(self.feature_names[idx]),
                    "mean_tfidf": round(float(mean_vec[idx]), 4),
                    "idf": round(float(self.idf_map[self.feature_names[idx]]), 4),
                }
                for idx in top_indices
                if mean_vec[idx] > 0
            ]
        return result
