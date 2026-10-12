# 📚 Intelligent Document Management & Search System (NLP & ML College Project)

An end-to-end **Natural Language Processing (NLP) & Machine Learning College Project** that ingests, preprocesses, vectorizes, classifies, clusters, and searches enterprise documents using **TF-IDF**, **Supervised Classification**, **Unsupervised K-Means Clustering**, **Inverted-Index Keyword Search**, and **Cosine Similarity Search**.

---

## 📂 Project Structure

```text
intelligent-document-system/
├── .vscode/
│   ├── launch.json                              # One-click Run/Debug configs for VS Code (F5)
│   └── settings.json                            # Auto-selects .venv Python interpreter in VS Code
├── data/
│   └── documents_corpus.csv                     # 510-row corpus (500 valid docs + 5 anomalies + 5 unlabeled)
├── models/
│   └── document_intelligence.joblib             # Serialized leak-free Classifier, TF-IDF, K-Means & Search Engine
├── notebooks/
│   └── document_intelligence_analysis.ipynb     # Storytelling Jupyter Notebook (EDA, TF-IDF, ML, Clustering, Q&A)
├── reports/
│   ├── figures/                                 # High-resolution diagnostic & clustering visualizations
│   │   ├── eda_distributions.png
│   │   ├── tfidf_top_terms.png
│   │   ├── classification_comparison.png
│   │   ├── confusion_matrices.png
│   │   └── clustering_silhouette_and_pca.png
│   └── metrics_summary.json                     # Complete evaluation metrics, 95% CIs, slice analysis & cluster profiles
├── src/
│   ├── __init__.py
│   ├── generate_corpus.py                       # Builds synthetic 5-domain document corpus with edge cases
│   ├── text_preprocessing.py                    # Function 1: NLP Document Preprocessing & Leak-Free ColumnTransformer
│   ├── tfidf_engine.py                          # Function 2: TF-IDF Vectorizer, Term Weights & Inverted Index
│   ├── classifier_and_clustering.py             # Function 3: Supervised Classification & Unsupervised K-Means Clustering
│   └── search_engine.py                         # Function 4 & 5: Keyword Search & Cosine Similarity Search CLI/API
├── app.py                                       # Interactive 4-Tab Streamlit Web Dashboard for Live Classroom Demos
├── .gitignore                                   # Excludes .venv, __pycache__, and OS temporary files
├── requirements.txt                             # Locked Python dependencies
└── README.md                                    # Project documentation, VS Code guide & GitHub tutorial
```

---

## ⚙️ Core System Functions

1. **Function 1 — Document Preprocessing (`src/text_preprocessing.py`)**:
   - **Schema & Anomaly Audit**: Flags and drops 5 corrupted records (negative experience `-8.5`, negative citations `-42.0`, empty document text, extreme experience `145.0` yrs, invalid binary flag `9`) and 5 unlabeled records (`category = NaN`), retaining 500 valid documents.
   - **NLP Text Normalization**: Lowercasing, punctuation/digit stripping, English + domain boilerplate stopword removal, and human-readable morphological lemmatization.
   - **Leak-Free Feature Engineering**: Converts `published_date` into Unix timestamps (`published_timestamp`), preserves pre-encoded binary `is_verified` (`0`/`1`), encodes ordinal `access_tier` (`Public < Internal < Confidential < Restricted`) with `OrdinalEncoder`, encodes nominal `source_channel` with `OneHotEncoder`, and imputes missing numerical values (`author_experience_years`, `citation_count`) via median imputation + missingness indicators (`add_indicator=True`) scaled with `StandardScaler`.
2. **Function 2 — TF-IDF Vectorization (`src/tfidf_engine.py`)**:
   - Fits an 800-feature unigram/bigram `TfidfVectorizer` (`sublinear_tf=True`, $L_2$ normalization) strictly on the **Training Set (350 docs)** to prevent data leakage.
   - Computes per-document Term Frequency (TF), Inverse Document Frequency (IDF), and $L_2$-normalized TF-IDF weights.
3. **Function 3 — Document Classification & Clustering (`src/classifier_and_clustering.py`)**:
   - **Supervised Classification**: Trains and compares 7 models across 5 categories (`Finance`, `Healthcare`, `Legal`, `Science`, `Technology`) using a strict **70% Train / 15% Validation / 15% Test** split, 5-fold Stratified Cross-Validation, 95% Bootstrap Confidence Intervals, and subgroup slice analysis.
   - **Unsupervised Clustering**: Evaluates $K$-Means across $k \in \{2..8\}$ via **Silhouette Score**, identifies optimal $k^* = 6$ (`Silhouette = 0.5663`), and projects documents into 2D using **PCA** (`50.54%` explained variance).
4. **Function 4 — Keyword Search (`src/search_engine.py`)**:
   - Uses an **Inverted Index** mapping vocabulary unigrams and bigrams to document posting lists, with title match boosting, category/access-tier filtering, and Markdown keyword highlighting.
5. **Function 5 — Similarity Search (`src/search_engine.py`)**:
   - Computes **Cosine Similarity** ($\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$) over $L_2$-normalized TF-IDF vectors for both **Query-to-Document** and **Document-to-Document** retrieval.

---

## 📊 Model Evaluation & Clustering Summary

### Supervised Classification Comparison (70% Train / 15% Val / 15% Test)

| Model | Train Macro F1 | 5-Fold CV F1 | Val Accuracy | Val Macro F1 | Val ROC-AUC | Val PR-AUC | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Majority)** | 0.0667 | 0.0667 | 20.00% | 0.0667 | 0.5000 | 0.2000 | 1.08 KB |
| **Decision Tree (Unregularized)** | 1.0000 | 0.7069 | 60.00% | 0.6014 | 0.7500 | 0.4471 | 43.84 KB |
| **Decision Tree (Regularized)** | 0.8572 | 0.6339 | 61.33% | 0.6141 | 0.8038 | 0.5780 | 37.34 KB |
| **Complement Naive Bayes (TF-IDF)** | 0.8401 | 0.8390 | 72.00% | 0.7118 | 0.9162 | 0.7769 | 103.52 KB |
| **Random Forest (Regularized)** | 0.9914 | 0.8739 | 77.33% | 0.7708 | 0.9636 | 0.8873 | 1372.69 KB |
| **Calibrated Linear SVM (L2)** | 1.0000 | 0.8570 | 73.33% | 0.7267 | 0.9618 | 0.8943 | 105.37 KB |
| **Logistic Regression (L2)** ⭐ | **1.0000** | **0.8536** | **73.33%** | **0.7267** | **0.9587** | **0.8836** | **55.21 KB** |

- **Held-Out Test Set Performance (`Logistic Regression L2`)**:
  - **Test Macro ROC-AUC:** `0.9833` (95% Bootstrap CI: `[0.9651, 0.9957]`)
  - **Test Macro PR-AUC:** `0.9449`
  - **Test Accuracy:** `85.33%` (95% Bootstrap CI: `[77.33%, 92.00%]`)
  - **Test Macro F1-Score:** `0.8545` (95% Bootstrap CI: `[0.7724, 0.9203]`)

### Unsupervised K-Means Clustering Profiles ($k^* = 6$, Silhouette Score = `0.5663`)

| Cluster ID | Docs | Dominant Category | Purity (%) | Top Discriminative TF-IDF Terms | Typical Tier & Channel |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **Cluster 0** | 100 | `Healthcare` | 81.00% | `clinical, patient, neurodegenerative, cerebrospinal, oncology` | Confidential / Research_Portal |
| **Cluster 1** | 104 | `Legal` | 78.85% | `compliance, regulatory, data, gdpr, privacy, environmental` | Confidential / Enterprise_Wiki |
| **Cluster 2** | 68 | `Science (Earth/Plasma)` | 79.41% | `ocean, tokamak, atmospheric, vent, sea, circulation` | Public / Research_Portal |
| **Cluster 3** | 92 | `Technology` | 83.70% | `learn, zero, distribut, database, edge, cybersecurity` | Internal / Enterprise_Wiki |
| **Cluster 4** | 98 | `Finance` | 85.71% | `central, adequacy, equity, portfolio, foreign, quantitative` | Confidential / Regulatory_Filing |
| **Cluster 5** | 38 | `Science (Quantum/Solar)` | 81.58% | `solar, cell, perovskite, error, correction, efficiency` | Public / Research_Portal |

---

## 💻 Step-by-Step Guide: Open, Edit & Run in VS Code

### Step 1: Open the Project Folder in VS Code
1. Open **Visual Studio Code**.
2. Click **File → Open Folder...** and select:
   `/Users/histoi/.gemini/antigravity/scratch/intelligent-document-system`
   *(Or run `code /Users/histoi/.gemini/antigravity/scratch/intelligent-document-system` in your terminal).*
3. Notice that `.vscode/settings.json` automatically configures VS Code to use the local `.venv/bin/python` virtual environment.

### Step 2: Run the Project via VS Code Terminal (`Ctrl+\``) or Run & Debug (`F5`)
Open the integrated terminal in VS Code (**Terminal → New Terminal**) and run:

```bash
# 1. (Optional) Regenerate the 510-document corpus
.venv/bin/python -m src.generate_corpus

# 2. Run NLP preprocessing, TF-IDF, Classification, Clustering & Plot Generation
.venv/bin/python -m src.classifier_and_clustering

# 3. Test Keyword Search & Cosine Similarity Search in the Terminal
.venv/bin/python -m src.search_engine --query "neural network transformer GPU inference" --top_k 5

# 4. Launch the Interactive 4-Tab Streamlit Web Application
.venv/bin/streamlit run app.py
```

Or press **`Cmd+Shift+D`** (Run and Debug panel in VS Code), choose any of the 3 pre-configured launch options (`1. Build Corpus & Train NLP/ML Models`, `2. Run Keyword & Similarity Search CLI Demo`, or `3. Launch Streamlit Intelligent Document System`), and press **`F5`**.

### Step 3: Open and Run the Jupyter Notebook in VS Code
1. In the VS Code Explorer sidebar, open `notebooks/document_intelligence_analysis.ipynb`.
2. In the top-right corner of the notebook editor, click **Select Kernel** and choose **`.venv (Python 3.14.7)`**.
3. Click **Run All** at the top of the notebook to execute all cells and view the inline tables and visualizations.

---

## 🚀 Step-by-Step Guide: Push Your Project to GitHub

A local Git repository is already initialized and committed in this folder. Follow these steps to push it to your GitHub account:

### Step 1: Create a New Repository on GitHub
1. Go to [https://github.com/new](https://github.com/new) and sign in.
2. Enter **Repository name**: `intelligent-document-system`
3. Choose **Public** (or Private).
4. **Important**: Leave *"Add a README file"*, *"Add .gitignore"*, and *"Choose a license"* **UNCHECKED** (since our project already includes them).
5. Click **Create repository**.

### Step 2: Link and Push from VS Code Terminal
Copy and run these 3 commands inside your VS Code terminal (replace `<YOUR_GITHUB_USERNAME>` with your actual GitHub username):

```bash
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/intelligent-document-system.git
git push -u origin main
```
*(If GitHub prompts for authentication on macOS, sign in via the browser pop-up or paste a GitHub Personal Access Token as your password).*
