"""
Module 4 & 5: Keyword Search (Inverted Index + TF-IDF Ranking) and
Semantic/Lexical Similarity Search (Cosine Similarity over TF-IDF Space).

Can be imported as a Python API or executed directly via CLI:
    .venv/bin/python -m src.search_engine --query "neural network transformer GPU" --top_k 5
"""

import argparse
import re
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.text_preprocessing import preprocess_text


def highlight_matching_terms(text: str, query_tokens: list[str]) -> str:
    """Highlights matching query tokens or stems in a document snippet using Markdown bolding."""
    if not query_tokens:
        return text
    highlighted = text
    for tok in sorted(set(query_tokens), key=len, reverse=True):
        if len(tok) < 2:
            continue
        pattern = re.compile(rf"\b({re.escape(tok)}\w*)\b", flags=re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)
    return highlighted


class DocumentSearchEngine:
    """
    Provides dual retrieval engines over the indexed document corpus:
    1. Keyword Search: Inverted Index lookup + exact title/content frequency boost + term highlighting
    2. Similarity Search: Cosine Similarity over L2-normalized TF-IDF vectors (Query-to-Doc and Doc-to-Doc)
    """

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        tfidf_engine: Any,
        cluster_labels: np.ndarray | list[int] | None = None,
    ):
        self.corpus_df = corpus_df.reset_index(drop=True).copy()
        if cluster_labels is not None:
            self.corpus_df["cluster_id"] = np.asarray(cluster_labels, dtype=int)
        elif "cluster_id" not in self.corpus_df.columns:
            self.corpus_df["cluster_id"] = 0

        self.tfidf_engine = tfidf_engine
        if self.tfidf_engine.corpus_tfidf_matrix is None:
            self.tfidf_engine.index_corpus(self.corpus_df["clean_text"])
        self.tfidf_matrix = self.tfidf_engine.corpus_tfidf_matrix

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: str | None = None,
        access_tier_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Executes Inverted-Index Keyword Search with exact & lemmatized token matching,
        title match boosting, metadata filtering, and highlighted snippets.
        """
        clean_query = preprocess_text(query)
        query_tokens = clean_query.split()
        raw_tokens = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in query.split() if len(w) >= 2]
        all_search_tokens = list(dict.fromkeys(query_tokens + raw_tokens))

        if not all_search_tokens:
            return []

        doc_scores: dict[int, float] = {}
        doc_matched_terms: dict[int, set[str]] = {}

        # 1. Inverted index lookup for unigrams and bigrams
        candidate_ngrams = list(all_search_tokens)
        for i in range(len(query_tokens) - 1):
            candidate_ngrams.append(f"{query_tokens[i]} {query_tokens[i + 1]}")

        for term in candidate_ngrams:
            postings = self.tfidf_engine.inverted_index.get(term, [])
            boost = 1.5 if " " in term else 1.0  # Bigram phrase bonus
            for doc_idx, weight in postings:
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + (weight * boost)
                doc_matched_terms.setdefault(doc_idx, set()).add(term)

        # 2. Substring/prefix fallback & title match boost across corpus
        for idx, row in self.corpus_df.iterrows():
            doc_idx = int(idx)
            title_lower = str(row["title"]).lower()
            clean_doc = str(row["clean_text"]).lower()

            for tok in all_search_tokens:
                if tok in title_lower:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + 0.35
                    doc_matched_terms.setdefault(doc_idx, set()).add(tok)
                elif tok in clean_doc and doc_idx not in doc_scores:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + 0.15
                    doc_matched_terms.setdefault(doc_idx, set()).add(tok)

        # 3. Filter and rank candidate documents
        ranked_indices = sorted(doc_scores.keys(), key=lambda i: doc_scores[i], reverse=True)
        results = []
        for doc_idx in ranked_indices:
            row = self.corpus_df.iloc[doc_idx]
            if category_filter and category_filter != "All" and row["category"] != category_filter:
                continue
            if access_tier_filter and access_tier_filter != "All" and row["access_tier"] != access_tier_filter:
                continue

            matched_list = sorted(doc_matched_terms.get(doc_idx, set()))
            coverage_ratio = len(matched_list) / max(len(set(query_tokens)), 1)
            final_score = float(doc_scores[doc_idx] * (1.0 + 0.25 * min(coverage_ratio, 1.0)))

            results.append(
                {
                    "rank": len(results) + 1,
                    "doc_id": str(row["doc_id"]),
                    "title": str(row["title"]),
                    "category": str(row["category"]),
                    "cluster_id": int(row["cluster_id"]),
                    "access_tier": str(row["access_tier"]),
                    "source_channel": str(row["source_channel"]),
                    "published_date": str(row["published_date"]),
                    "keyword_score": round(final_score, 4),
                    "matched_keywords": matched_list,
                    "highlighted_snippet": highlight_matching_terms(str(row["content"]), all_search_tokens),
                    "content": str(row["content"]),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Computes Cosine Similarity between a free-text query vector and all
        document TF-IDF vectors in the corpus, returning the top-K closest documents.
        """
        query_vec = self.tfidf_engine.transform_text(query_text, is_preprocessed=False).reshape(1, -1)
        if np.all(query_vec == 0):
            return []

        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        sorted_indices = np.argsort(sims)[::-1]

        q_non_zero = set(np.where(query_vec[0] > 0)[0])
        results = []

        for idx in sorted_indices:
            sim_score = float(sims[idx])
            if sim_score <= 0.0:
                break
            row = self.corpus_df.iloc[int(idx)]
            if category_filter and category_filter != "All" and row["category"] != category_filter:
                continue

            doc_vec = self.tfidf_matrix[int(idx)]
            d_non_zero = set(np.where(doc_vec > 0)[0])
            shared_idx = sorted(
                q_non_zero.intersection(d_non_zero),
                key=lambda i: float(query_vec[0, i] * doc_vec[i]),
                reverse=True,
            )
            shared_terms = [str(self.tfidf_engine.feature_names[i]) for i in shared_idx[:8]]

            results.append(
                {
                    "rank": len(results) + 1,
                    "doc_id": str(row["doc_id"]),
                    "title": str(row["title"]),
                    "category": str(row["category"]),
                    "cluster_id": int(row["cluster_id"]),
                    "access_tier": str(row["access_tier"]),
                    "source_channel": str(row["source_channel"]),
                    "published_date": str(row["published_date"]),
                    "cosine_similarity": round(sim_score, 4),
                    "similarity_percentage": f"{sim_score * 100:.2f}%",
                    "shared_tfidf_terms": shared_terms,
                    "content": str(row["content"]),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def find_similar_documents(self, doc_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Finds the top-K most similar documents in the corpus to an existing `doc_id`."""
        matches = self.corpus_df.index[self.corpus_df["doc_id"] == doc_id].tolist()
        if not matches:
            raise ValueError(f"Document ID '{doc_id}' not found in corpus.")

        target_idx = int(matches[0])
        target_vec = self.tfidf_matrix[target_idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.tfidf_matrix)[0]

        sorted_indices = np.argsort(sims)[::-1]
        t_non_zero = set(np.where(target_vec[0] > 0)[0])
        results = []

        for idx in sorted_indices:
            if int(idx) == target_idx:
                continue
            sim_score = float(sims[idx])
            row = self.corpus_df.iloc[int(idx)]
            doc_vec = self.tfidf_matrix[int(idx)]
            d_non_zero = set(np.where(doc_vec > 0)[0])
            shared_idx = sorted(
                t_non_zero.intersection(d_non_zero),
                key=lambda i: float(target_vec[0, i] * doc_vec[i]),
                reverse=True,
            )
            shared_terms = [str(self.tfidf_engine.feature_names[i]) for i in shared_idx[:8]]

            results.append(
                {
                    "rank": len(results) + 1,
                    "doc_id": str(row["doc_id"]),
                    "title": str(row["title"]),
                    "category": str(row["category"]),
                    "cluster_id": int(row["cluster_id"]),
                    "access_tier": str(row["access_tier"]),
                    "cosine_similarity": round(sim_score, 4),
                    "similarity_percentage": f"{sim_score * 100:.2f}%",
                    "shared_tfidf_terms": shared_terms,
                    "content": str(row["content"]),
                }
            )
            if len(results) >= top_k:
                break

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent Document Keyword & Similarity Search CLI")
    parser.add_argument(
        "--query",
        type=str,
        default="neural network transformer deep learning GPU inference",
        help="Search query string",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--doc_id", type=str, default="DOC-1001", help="Document ID for Doc-to-Doc similarity")
    args = parser.parse_args()

    bundle_path = Path("models/document_intelligence.joblib")
    if not bundle_path.exists():
        print("Model bundle not found. Please run `python -m src.classifier_and_clustering` first.")
        return

    bundle = joblib.load(bundle_path)
    search_engine: DocumentSearchEngine = bundle["search_engine"]

    print("=" * 80)
    print(f"🔎 KEYWORD SEARCH RESULTS FOR: '{args.query}'")
    print("=" * 80)
    kw_results = search_engine.keyword_search(args.query, top_k=args.top_k)
    for res in kw_results:
        print(
            f"[{res['rank']}] {res['doc_id']} | {res['title']} "
            f"(Category: {res['category']} | Score: {res['keyword_score']:.4f})"
        )
        print(f"    Matched Terms: {', '.join(res['matched_keywords'])}")
        print(f"    Snippet: {res['highlighted_snippet'][:150]}...\n")

    print("=" * 80)
    print(f"🧭 COSINE SIMILARITY SEARCH RESULTS FOR: '{args.query}'")
    print("=" * 80)
    sim_results = search_engine.similarity_search(args.query, top_k=args.top_k)
    for res in sim_results:
        print(
            f"[{res['rank']}] {res['doc_id']} | {res['title']} "
            f"(Category: {res['category']} | Cosine Similarity: {res['similarity_percentage']})"
        )
        print(f"    Shared TF-IDF Terms: {', '.join(res['shared_tfidf_terms'])}\n")

    print("=" * 80)
    print(f"📄 DOCUMENT-TO-DOCUMENT SIMILARITY FOR: '{args.doc_id}'")
    print("=" * 80)
    doc_peers = search_engine.find_similar_documents(args.doc_id, top_k=3)
    for res in doc_peers:
        print(
            f"[{res['rank']}] {res['doc_id']} | {res['title']} "
            f"(Category: {res['category']} | Similarity: {res['similarity_percentage']})"
        )


if __name__ == "__main__":
    main()
