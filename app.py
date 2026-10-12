"""
Interactive Streamlit Web Application for the Intelligent Document Management and Search System.
Covers all 5 core modules:
1. NLP Document Preprocessing
2. TF-IDF Feature Extraction & Term Weight Inspection
3. Supervised Document Classification & Unsupervised K-Means Clustering
4. Inverted-Index Keyword Search
5. TF-IDF Cosine Similarity Search (Query-to-Doc & Doc-to-Doc)

Run locally via:
    .venv/bin/streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.classifier_and_clustering import analyze_single_document, train_and_evaluate_all


st.set_page_config(
    page_title="Intelligent Document Management & Search System | NLP & ML Project",
    page_icon="📚",
    layout="wide",
)

MODEL_PATH = Path("models/document_intelligence.joblib")
METRICS_PATH = Path("reports/metrics_summary.json")


@st.cache_resource
def load_system_bundle():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_evaluate_all()
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return bundle, metrics


bundle, metrics_report = load_system_bundle()
search_engine = bundle["search_engine"]

st.title("📚 Intelligent Document Management & Search System")
st.markdown(
    """
    **College NLP & Machine Learning Project** — End-to-end system featuring **1. NLP Document Preprocessing**,
    **2. TF-IDF Vectorization**, **3. Supervised Document Classification**, **4. Unsupervised K-Means Clustering**,
    **5. Inverted-Index Keyword Search**, and **6. Cosine Similarity Search**.
    """
)

tab_nlp_tfidf, tab_clf_cluster, tab_search, tab_corpus = st.tabs(
    [
        "🧹 1. NLP Preprocessing & TF-IDF",
        "🏷️ 2. Classification & Clustering",
        "🔎 3. Keyword & Similarity Search",
        "🗂️ 4. Document Corpus Manager",
    ]
)

SAMPLE_DOCS = {
    "Technology (AI & Transformers)": (
        "Transformer Neural Network GPU Inference Optimization",
        "This technical study evaluates deep learning transformer attention mechanisms and GPU tensor quantization "
        "to reduce neural network inference latency in distributed Kubernetes cloud clusters.",
        "Internal",
        "Enterprise_Wiki",
        8.5,
        34.0,
    ),
    "Finance (Monetary Policy & Risk)": (
        "Central Bank Yield Curve and Portfolio Credit Risk",
        "Macroeconomic analysis of central bank monetary policy, treasury bond yield curve inversions, "
        "quantitative tightening liquidity, and Basel III credit risk capital adequacy stress testing.",
        "Confidential",
        "Regulatory_Filing",
        11.0,
        22.0,
    ),
    "Healthcare (Clinical Oncology Trial)": (
        "Phase III Immunotherapy Biomarker Clinical Trial",
        "Randomized double blind placebo controlled oncology clinical trial evaluating patient survival, "
        "genomic sequencing biomarkers, and adverse event dosage efficacy in hospital oncology wards.",
        "Confidential",
        "Research_Portal",
        14.0,
        52.0,
    ),
    "Legal (GDPR Privacy & Patent Law)": (
        "Cross-Border GDPR Data Privacy and Patent Litigation",
        "Regulatory compliance opinion regarding GDPR personal identifiable information consent, data controller "
        "cross border transfer liability, software patent infringement prior art, and statutory damages.",
        "Restricted",
        "Regulatory_Filing",
        15.0,
        14.0,
    ),
    "Science (Astrophysics & Quantum)": (
        "Exoplanet Atmospheric Spectroscopy and Qubit Coherence",
        "Astrophysics infrared space telescope spectroscopy of exoplanet atmospheric methane transits alongside "
        "superconducting qubit quantum error correction and cryogenic coherence benchmarks.",
        "Public",
        "Research_Portal",
        12.0,
        64.0,
    ),
}

with st.sidebar:
    st.header("📄 Active Document Input")
    preset_choice = st.selectbox("Load Sample Document Preset", list(SAMPLE_DOCS.keys()))
    default_title, default_body, def_tier, def_chan, def_exp, def_cit = SAMPLE_DOCS[preset_choice]

    doc_title = st.text_input("Document Title", value=default_title)
    doc_body = st.text_area("Document Content", value=default_body, height=145)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        access_tier = st.selectbox(
            "Access Tier (Ordinal)",
            ["Public", "Internal", "Confidential", "Restricted"],
            index=["Public", "Internal", "Confidential", "Restricted"].index(def_tier),
        )
        exp_years = st.number_input("Author Experience (Yrs)", min_value=0.0, max_value=50.0, value=def_exp, step=0.5)
    with col_s2:
        source_channel = st.selectbox(
            "Source Channel (Nominal)",
            ["Enterprise_Wiki", "Research_Portal", "News_Wire", "Regulatory_Filing"],
            index=["Enterprise_Wiki", "Research_Portal", "News_Wire", "Regulatory_Filing"].index(def_chan),
        )
        citations = st.number_input("Citation Count", min_value=0.0, max_value=500.0, value=def_cit, step=1.0)
    is_verified = st.checkbox("Verified Document (Pre-encoded Binary = 1)", value=True)

analysis = analyze_single_document(
    title=doc_title,
    content=doc_body,
    access_tier=access_tier,
    source_channel=source_channel,
    author_experience_years=exp_years,
    citation_count=citations,
    is_verified=1 if is_verified else 0,
)

# ---------------------------------------------------------------------------
# TAB 1: NLP PREPROCESSING & TF-IDF
# ---------------------------------------------------------------------------
with tab_nlp_tfidf:
    st.subheader("1️⃣ NLP Document Preprocessing Pipeline")
    stats = analysis["nlp_preprocessing_stats"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw Word Count", stats["raw_word_count"])
    m2.metric("Clean Lemmatized Tokens", stats["clean_token_count"])
    m3.metric("Unique Vocabulary Terms", stats["unique_token_count"])
    m4.metric("Lexical Diversity Ratio", f"{stats['lexical_diversity']:.2f}")

    c_raw, c_clean = st.columns(2)
    with c_raw:
        st.markdown("**Original Raw Text (`Title + Content`):**")
        st.info(f"**{doc_title}**. {doc_body}")
    with c_clean:
        st.markdown("**Preprocessed & Lemmatized Text (Stopwords & Punctuation Removed):**")
        st.success(stats["preprocessed_text"] or "(No valid tokens found)")

    st.divider()
    st.subheader("2️⃣ TF-IDF Vectorization & Term Weight Breakdown")
    st.markdown(
        r"Each term $t$ in document $d$ is weighted by sublinear Term Frequency $\text{tf}(t,d) = 1 + \ln f_{t,d}$ "
        r"multiplied by Inverse Document Frequency $\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$, "
        r"followed by $L_2$ unit vector normalization."
    )
    tfidf_df = pd.DataFrame(analysis["tfidf_term_table"])
    if not tfidf_df.empty:
        col_tbl, col_chart = st.columns([1.25, 1])
        with col_tbl:
            st.dataframe(tfidf_df, use_container_width=True)
        with col_chart:
            st.bar_chart(tfidf_df.set_index("term")["tfidf_weight"], horizontal=True)
    else:
        st.warning("No matching vocabulary terms found in the trained TF-IDF dictionary.")

    fig_tfidf = Path("reports/figures/tfidf_top_terms.png")
    if fig_tfidf.exists():
        st.markdown("#### Corpus-Wide Top Discriminative TF-IDF Terms by Category")
        st.image(str(fig_tfidf), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: CLASSIFICATION & CLUSTERING
# ---------------------------------------------------------------------------
with tab_clf_cluster:
    st.subheader("3️⃣ Automatic Document Classification & K-Means Cluster Assignment")
    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted Category (Supervised)", analysis["predicted_category"])
    k2.metric("Classification Confidence", analysis["confidence_percentage"])
    k3.metric("Assigned Topic Cluster (Unsupervised)", f"Cluster {analysis['assigned_cluster_id']}")

    col_prob, col_cprof = st.columns([1, 1.2])
    with col_prob:
        st.markdown("**Predicted Category Probabilities (%):**")
        prob_df = pd.DataFrame(
            list(analysis["class_probabilities"].items()),
            columns=["Category", "Probability (%)"],
        ).sort_values(by="Probability (%)", ascending=False)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with col_cprof:
        st.markdown(f"**Assigned K-Means Cluster Profile (`Cluster {analysis['assigned_cluster_id']}`):**")
        c_prof = analysis["assigned_cluster_profile"]
        if c_prof:
            st.write(
                f"- **Dominant Category:** `{c_prof['dominant_category']}` "
                f"({c_prof['dominant_category_purity_pct']}% cluster purity across {c_prof['document_count']} docs)\n"
                f"- **Top Cluster TF-IDF Terms:** `{', '.join(c_prof['top_tfidf_terms'])}`\n"
                f"- **Typical Access Tier & Channel:** `{c_prof['typical_access_tier']}` via `{c_prof['typical_source_channel']}`\n"
                f"- **Mean Citations / Experience:** `{c_prof['mean_citation_count']}` citations | `{c_prof['mean_author_experience_years']}` yrs"
            )

    st.divider()
    st.subheader("📊 Supervised Classification & Unsupervised Clustering Benchmarks")
    test_eval = metrics_report["final_test_evaluation"]
    clust_eval = metrics_report["clustering_evaluation"]

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Test Macro F1 (Logistic Reg L2)", f"{test_eval['test_macro_f1']:.4f}")
    b2.metric("Test Macro ROC-AUC", f"{test_eval['test_roc_auc']:.4f}")
    b3.metric("Optimal K-Means Clusters (k*)", f"k = {clust_eval['optimal_k']}")
    b4.metric("Optimal Silhouette Score", f"{clust_eval['optimal_silhouette_score']:.4f}")

    comp_df = pd.DataFrame(metrics_report["validation_model_comparison"]).T.drop(columns=["confusion_matrix"])
    st.markdown("#### Multi-Model Validation Comparison (Leak-Free 70/15/15 Split)")
    st.dataframe(comp_df, use_container_width=True)

    fig_dir = Path("reports/figures")
    if (fig_dir / "classification_comparison.png").exists():
        st.image(str(fig_dir / "classification_comparison.png"), caption="Model F1 Comparison, ROC Curves, and Calibration")
    if (fig_dir / "confusion_matrices.png").exists():
        st.image(str(fig_dir / "confusion_matrices.png"), caption="Overfitting Remediation & Held-Out Test Confusion Matrices")
    if (fig_dir / "clustering_silhouette_and_pca.png").exists():
        st.image(
            str(fig_dir / "clustering_silhouette_and_pca.png"),
            caption="K-Means Silhouette Optimization & 2D PCA Projections (Cluster Hue vs. Discrete Category Hue)",
        )

# ---------------------------------------------------------------------------
# TAB 3: KEYWORD SEARCH & COSINE SIMILARITY SEARCH
# ---------------------------------------------------------------------------
with tab_search:
    st.subheader("4️⃣ Inverted-Index Keyword Search & 5️⃣ Cosine Similarity Search")

    sc1, sc2, sc3, sc4 = st.columns([2.2, 1, 1, 0.8])
    with sc1:
        search_query = st.text_input(
            "Enter Search Query or Natural Language Topic",
            value="transformer neural network GPU latency optimization",
        )
    with sc2:
        cat_filter = st.selectbox(
            "Filter by Category",
            ["All", "Technology", "Finance", "Healthcare", "Legal", "Science"],
        )
    with sc3:
        tier_filter = st.selectbox(
            "Filter by Access Tier",
            ["All", "Public", "Internal", "Confidential", "Restricted"],
        )
    with sc4:
        top_k = st.slider("Top-K Results", min_value=3, max_value=15, value=5)

    col_kw, col_sim = st.columns(2)

    with col_kw:
        st.markdown("### 🔑 Keyword Search Results (Inverted Index)")
        kw_hits = search_engine.keyword_search(
            query=search_query,
            top_k=top_k,
            category_filter=cat_filter,
            access_tier_filter=tier_filter,
        )
        if not kw_hits:
            st.info("No exact or stemmed keyword matches found for this query and filter combination.")
        for item in kw_hits:
            with st.container(border=True):
                st.markdown(
                    f"**#{item['rank']} [{item['doc_id']}] {item['title']}**  \n"
                    f"🏷️ **Category:** `{item['category']}` | 🧩 **Cluster:** `Cluster {item['cluster_id']}` | "
                    f"🔒 **Tier:** `{item['access_tier']}` | 📈 **Keyword Score:** `{item['keyword_score']:.4f}`"
                )
                st.caption(f"Matched Tokens: `{', '.join(item['matched_keywords'])}`")
                st.markdown(item["highlighted_snippet"])

    with col_sim:
        st.markdown("### 🧭 Cosine Similarity Search Results (TF-IDF Vector Space)")
        sim_hits = search_engine.similarity_search(
            query_text=search_query,
            top_k=top_k,
            category_filter=cat_filter,
        )
        if not sim_hits:
            st.info("No documents had non-zero cosine similarity with the query.")
        for item in sim_hits:
            with st.container(border=True):
                st.markdown(
                    f"**#{item['rank']} [{item['doc_id']}] {item['title']}**  \n"
                    f"🏷️ **Category:** `{item['category']}` | 🧩 **Cluster:** `Cluster {item['cluster_id']}` | "
                    f"🎯 **Cosine Similarity:** `{item['similarity_percentage']}` (`{item['cosine_similarity']:.4f}`)"
                )
                st.caption(f"Shared High-Weight TF-IDF Terms: `{', '.join(item['shared_tfidf_terms'])}`")
                st.write(item["content"])

    st.divider()
    st.markdown("### 📄 Document-to-Document Similarity Finder")
    all_doc_ids = search_engine.corpus_df["doc_id"].tolist()
    selected_doc_id = st.selectbox("Select an Existing Document ID to Find Similar Peers", all_doc_ids[:50])
    target_row = search_engine.corpus_df[search_engine.corpus_df["doc_id"] == selected_doc_id].iloc[0]
    st.info(
        f"**Selected Document (`{selected_doc_id}`):** {target_row['title']} — "
        f"*Category: `{target_row['category']}` | Cluster: `Cluster {target_row['cluster_id']}`*\n\n"
        f"{target_row['content']}"
    )
    peer_docs = search_engine.find_similar_documents(selected_doc_id, top_k=4)
    st.dataframe(
        pd.DataFrame(peer_docs)[
            ["rank", "doc_id", "title", "category", "cluster_id", "similarity_percentage", "shared_tfidf_terms"]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# TAB 4: DOCUMENT CORPUS MANAGER
# ---------------------------------------------------------------------------
with tab_corpus:
    st.subheader("🗂️ Enterprise Document Corpus Explorer & Audit Summary")
    audit = metrics_report["data_hygiene_and_schema_audit"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Raw Corpus Rows", audit["initial_row_count"])
    a2.metric("Anomalous Rows Dropped", audit["anomalous_row_count"])
    a3.metric("Unlabeled Target Rows Dropped", audit["missing_target_row_count"])
    a4.metric("Clean Indexed Documents", audit["clean_row_count"])

    display_df = search_engine.corpus_df[
        [
            "doc_id",
            "title",
            "category",
            "cluster_id",
            "access_tier",
            "source_channel",
            "published_date",
            "author_experience_years",
            "citation_count",
            "word_count",
        ]
    ]
    st.dataframe(display_df, use_container_width=True, height=380)

    csv_path = Path("data/documents_corpus.csv")
    if csv_path.exists():
        st.download_button(
            "⬇️ Download Full Document Corpus CSV",
            data=csv_path.read_bytes(),
            file_name="documents_corpus.csv",
            mime="text/csv",
        )
