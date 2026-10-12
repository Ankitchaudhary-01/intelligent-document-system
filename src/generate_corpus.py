"""
Generates a realistic 5-domain enterprise document corpus (`data/documents_corpus.csv`)
with rich text and structured metadata for NLP Preprocessing, TF-IDF Vectorization,
Document Classification, K-Means Clustering, Keyword Search, and Cosine Similarity Search.

Includes:
- 500 valid documents across 5 categories:
  1. Technology (AI, Cloud Computing, Cybersecurity, Distributed Systems)
  2. Finance (Monetary Policy, Equity Markets, Risk Management, Algorithmic Trading)
  3. Healthcare (Clinical Trials, Genomics, Cardiology, Medical Imaging)
  4. Legal (Intellectual Property, Data Privacy GDPR, Corporate Governance, Antitrust)
  5. Science (Astrophysics, Quantum Mechanics, Climate Modeling, Renewable Energy)
- 5 anomalous rows that violate schema constraints (negative citations, negative experience, empty text)
- 5 rows with missing target values (`category` is NaN)
- Realistic missing values (~6%) in `author_experience_years` and `citation_count`
  to demonstrate leak-free imputation and missingness indicators.
"""

from pathlib import Path
import numpy as np
import pandas as pd


DOMAIN_TEMPLATES = {
    "Technology": {
        "topics": [
            ("Transformer Neural Network Optimization", "deep learning transformer attention mechanism GPU tensor inference quantization neural network latency throughput model weights"),
            ("Kubernetes Microservices Orchestration", "cloud native container docker kubernetes cluster autoscaling service mesh ingress envoy load balancing distributed telemetry"),
            ("Zero Trust Cybersecurity Architecture", "cybersecurity zero trust encryption authentication firewall intrusion detection endpoint security vulnerability cryptographic token"),
            ("Distributed Database Consensus Protocols", "distributed database raft paxos replication sharding eventual consistency low latency write ahead log fault tolerance storage"),
            ("Edge AI and Federated Learning Systems", "federated learning edge computing iot sensor neural network privacy preserving gradient aggregation bandwidth optimization firmware"),
            ("Large Language Model Retrieval Augmented Generation", "retrieval augmented generation vector database embedding semantic search prompt engineering transformer token context window"),
        ],
        "channels": ["Enterprise_Wiki", "Research_Portal", "News_Wire"],
        "access_weights": [0.35, 0.40, 0.20, 0.05],  # Public, Internal, Confidential, Restricted
        "exp_mean": 7.5,
        "cit_mean": 28.0,
    },
    "Finance": {
        "topics": [
            ("Central Bank Monetary Policy and Yield Curves", "monetary policy interest rate inflation central bank treasury bond yield curve liquidity quantitative tightening macroeconomic"),
            ("Quantitative Equity Portfolio Optimization", "portfolio optimization sharpe ratio volatility equity factorinvesting asset allocation covariance matrix drawdown hedging alpha"),
            ("Basel III Capital Adequacy and Credit Risk", "credit risk basel capital adequacy stress testing default probability counterparty exposure balance sheet regulatory liquidity"),
            ("High Frequency Algorithmic Market Making", "algorithmic trading order book market making bid ask spread execution latency arbitrage derivatives futures clearing house"),
            ("Corporate Mergers Valuation and Cash Flow", "discounted cash flow enterprise valuation ebitda merger acquisition balance sheet debt equity leverage dividend payout"),
            ("Foreign Exchange Derivatives and Currency Swaps", "foreign exchange currency swap interest rate parity hedging volatility spot market forward contract central clearing"),
        ],
        "channels": ["Regulatory_Filing", "News_Wire", "Enterprise_Wiki"],
        "access_weights": [0.20, 0.35, 0.35, 0.10],
        "exp_mean": 10.5,
        "cit_mean": 18.0,
    },
    "Healthcare": {
        "topics": [
            ("Phase III Oncology Immunotherapy Clinical Trial", "clinical trial immunotherapy oncology patient survival biomarker randomized double blind placebo dosage adverse event efficacy"),
            ("CRISPR Gene Editing and Genomic Sequencing", "genomic sequencing crispr gene editing mutation dna transcription hereditary therapy cellular pathology molecular diagnostics"),
            ("Cardiovascular Arrhythmia Diagnostic Imaging", "cardiology electrocardiogram arrhythmia myocardial infarction echocardiography blood pressure vascular stent diagnostic imaging"),
            ("Hospital Epidemiology and Antimicrobial Resistance", "epidemiology antimicrobial resistance antibiotic pathogen infection control clinical protocol intensive care vaccine immunization"),
            ("Neurodegenerative Biomarkers in Cerebrospinal Fluid", "neurology neurodegenerative cognitive biomarker cerebrospinal fluid mri scan synaptic plasticity clinical progression therapy"),
            ("Telemedicine Remote Patient Monitoring Systems", "telemedicine wearable biosensor remote patient monitoring chronic disease glycemic blood glucose clinical alertehr integration"),
        ],
        "channels": ["Research_Portal", "Regulatory_Filing", "Enterprise_Wiki"],
        "access_weights": [0.25, 0.25, 0.35, 0.15],
        "exp_mean": 13.0,
        "cit_mean": 45.0,
    },
    "Legal": {
        "topics": [
            ("GDPR Cross-Border Data Privacy Compliance", "data privacy gdpr compliance personal identifiable information consent data controller processor cross border transfer regulatory fine"),
            ("Software Patent Litigation and Prior Art", "intellectual property patent infringement claims prior art licensing injunction federal circuit statutory damages trade secret"),
            ("Corporate Fiduciary Duty and Shareholder Governance", "corporate governance board of directors fiduciary duty shareholder derivative litigation proxy disclosure bylaws Delaware law"),
            ("Antitrust Market Concentration and Merger Review", "antitrust competition monopoly market concentration horizontal merger regulatory scrutiny consumer welfare consent decree injunction"),
            ("Commercial SaaS Master Service Agreement Liability", "contract law indemnity limitation of liability service level agreement arbitration clause breach of warranty intellectual property"),
            ("Environmental Regulatory Permitting and Clean Air", "environmental law regulatory compliance emissions permit administrative tribunal statutory enforcement remediation penalty"),
        ],
        "channels": ["Regulatory_Filing", "Enterprise_Wiki", "News_Wire"],
        "access_weights": [0.15, 0.25, 0.35, 0.25],
        "exp_mean": 14.5,
        "cit_mean": 12.0,
    },
    "Science": {
        "topics": [
            ("Exoplanet Atmospheric Spectroscopy via Space Telescope", "astrophysics exoplanet spectroscopy infrared telescope stellar transit atmospheric methane orbital mechanics luminosity redshift"),
            ("Superconducting Qubit Decoherence Error Correction", "quantum computing superconducting qubit coherence quantum error correction entanglement cryogenicjosephson junction gate fidelity"),
            ("Global Ocean Thermohaline Circulation Climate Model", "climate modeling ocean circulation atmospheric carbon dioxide greenhouse radiative forcing sea surface temperature paleoclimate"),
            ("Perovskite Tandem Photovoltaic Solar Cell Efficiency", "renewable energy perovskite photovoltaic solar cell bandgap photon conversion efficiency semiconductor thin film degradation"),
            ("Tokamak Magnetic Confinement Plasma Fusion", "nuclear fusion tokamak magnetic confinement plasma instability superconducting magnet tritium breeding thermal divertor"),
            ("Deep Sea Hydrothermal Vent Chemosynthetic Ecosystems", "marine biology hydrothermal vent chemosynthesis thermophilic archaea mineral precipitation ocean trench biogeochemical cycle"),
        ],
        "channels": ["Research_Portal", "News_Wire", "Enterprise_Wiki"],
        "access_weights": [0.55, 0.30, 0.10, 0.05],
        "exp_mean": 11.5,
        "cit_mean": 58.0,
    },
}

SHARED_CONNECTORS = [
    "In this comprehensive report, we evaluate systematic methodologies and empirical benchmarks across multiple operational environments.",
    "Furthermore, statistical analysis confirms significant improvements in reliability, scalability, and governance outcomes.",
    "Key stakeholders must review these findings to align strategic planning with long-term institutional objectives.",
    "Experimental results and longitudinal observations validate the theoretical framework established in prior literature.",
    "Additional sensitivity analyses highlight the importance of continuous monitoring and quality assurance protocols.",
]

ACCESS_TIERS = ["Public", "Internal", "Confidential", "Restricted"]


def generate_corpus(output_path: str | Path = "data/documents_corpus.csv", seed: int = 42) -> pd.DataFrame:
    """Generates the synthetic 510-row document corpus with realistic text, metadata, and edge cases."""
    rng = np.random.default_rng(seed)
    records = []
    doc_counter = 1001

    categories = list(DOMAIN_TEMPLATES.keys())
    docs_per_category = 100

    base_date = pd.Timestamp("2023-01-01")

    for category in categories:
        cfg = DOMAIN_TEMPLATES[category]
        topics = cfg["topics"]

        for i in range(docs_per_category):
            topic_idx = i % len(topics)
            base_title, core_vocab = topics[topic_idx]
            secondary_title, secondary_vocab = topics[(topic_idx + int(rng.integers(1, len(topics)))) % len(topics)]

            vocab_tokens = core_vocab.split()
            sec_tokens = secondary_vocab.split()

            # Sample domain words + realistic interdisciplinary cross-domain overlap
            other_cat_1 = categories[(categories.index(category) + int(rng.integers(1, len(categories)))) % len(categories)]
            other_cat_2 = categories[(categories.index(category) + int(rng.integers(2, len(categories)))) % len(categories)]
            noise_vocab_1 = DOMAIN_TEMPLATES[other_cat_1]["topics"][int(rng.integers(0, 6))][1].split()
            noise_vocab_2 = DOMAIN_TEMPLATES[other_cat_2]["topics"][int(rng.integers(0, 6))][1].split()

            # Occasionally create heavily interdisciplinary documents (~14% of corpus)
            if rng.random() < 0.14:
                chosen_core = rng.choice(vocab_tokens, size=4, replace=True)
                chosen_sec = rng.choice(sec_tokens, size=2, replace=True)
                chosen_noise = rng.choice(noise_vocab_1 + noise_vocab_2, size=10, replace=True)
                title_topic = DOMAIN_TEMPLATES[other_cat_1]["topics"][int(rng.integers(0, 6))][0]
            else:
                chosen_core = rng.choice(vocab_tokens, size=11, replace=True)
                chosen_sec = rng.choice(sec_tokens, size=5, replace=True)
                chosen_noise = rng.choice(noise_vocab_1 + noise_vocab_2, size=4, replace=False)
                title_topic = base_title

            connector_1, connector_2 = rng.choice(SHARED_CONNECTORS, size=2, replace=False)

            edition_num = int(rng.integers(1, 40))
            title = f"{title_topic} — Enterprise Report #{edition_num}"

            # Construct multi-sentence realistic document body
            sentence_1 = (
                f"This document investigates {title_topic.lower()} focusing on "
                f"{' '.join(chosen_core[:6])}."
            )
            sentence_2 = (
                f"Our methodology integrates {' '.join(chosen_core[6:])} alongside "
                f"{' '.join(chosen_sec[:3])} to address operational challenges."
            )
            sentence_3 = connector_1
            sentence_4 = (
                f"Comparative evaluation also considers {' '.join(chosen_sec[3:])} "
                f"and cross-functional aspects such as {' '.join(chosen_noise)}."
            )
            sentence_5 = connector_2

            content = f"{sentence_1} {sentence_2} {sentence_3} {sentence_4} {sentence_5}"

            # Metadata generation
            days_offset = int(rng.integers(0, 1000))
            pub_date = (base_date + pd.Timedelta(days=days_offset)).strftime("%Y-%m-%d")
            access_tier = rng.choice(ACCESS_TIERS, p=cfg["access_weights"])
            source_channel = rng.choice(cfg["channels"], p=[0.5, 0.3, 0.2])
            is_verified = int(rng.choice([1, 0], p=[0.82, 0.18]))

            exp_years = float(np.round(np.clip(rng.normal(cfg["exp_mean"], 3.2), 1.0, 35.0), 1))
            citations = float(np.round(np.clip(rng.normal(cfg["cit_mean"], 14.0), 0.0, 250.0), 0))

            # Introduce ~6% missingness in numerical metadata columns
            if rng.random() < 0.06:
                exp_years = np.nan
            if rng.random() < 0.06:
                citations = np.nan

            records.append(
                {
                    "doc_id": f"DOC-{doc_counter}",
                    "title": title,
                    "content": content,
                    "published_date": pub_date,
                    "access_tier": access_tier,
                    "source_channel": source_channel,
                    "is_verified": is_verified,
                    "author_experience_years": exp_years,
                    "citation_count": citations,
                    "category": category,
                }
            )
            doc_counter += 1

    # Add 5 anomalous rows that violate schema/domain constraints ("rows that don't make sense")
    anomalous_rows = [
        {
            "doc_id": "DOC-9001",
            "title": "Corrupted Record: Negative Author Experience",
            "content": "Neural network transformer GPU inference optimization study in distributed cloud clusters.",
            "published_date": "2024-05-10",
            "access_tier": "Internal",
            "source_channel": "Enterprise_Wiki",
            "is_verified": 1,
            "author_experience_years": -8.5,  # Impossible negative experience
            "citation_count": 15.0,
            "category": "Technology",
        },
        {
            "doc_id": "DOC-9002",
            "title": "Corrupted Record: Negative Citation Count",
            "content": "Monetary policy yield curve inflation central bank quantitative tightening analysis.",
            "published_date": "2024-06-12",
            "access_tier": "Confidential",
            "source_channel": "Regulatory_Filing",
            "is_verified": 1,
            "author_experience_years": 9.0,
            "citation_count": -42.0,  # Impossible negative citations
            "category": "Finance",
        },
        {
            "doc_id": "DOC-9003",
            "title": "Corrupted Record: Empty Document Body",
            "content": "   ",  # Empty / whitespace-only document content
            "published_date": "2024-07-01",
            "access_tier": "Public",
            "source_channel": "Research_Portal",
            "is_verified": 0,
            "author_experience_years": 5.0,
            "citation_count": 10.0,
            "category": "Healthcare",
        },
        {
            "doc_id": "DOC-9004",
            "title": "Corrupted Record: Extreme Author Experience (>100 yrs)",
            "content": "GDPR data privacy cross border personal identifiable information regulatory compliance.",
            "published_date": "2024-08-19",
            "access_tier": "Restricted",
            "source_channel": "Regulatory_Filing",
            "is_verified": 1,
            "author_experience_years": 145.0,  # Impossible human career experience
            "citation_count": 8.0,
            "category": "Legal",
        },
        {
            "doc_id": "DOC-9005",
            "title": "Corrupted Record: Invalid Verification Flag",
            "content": "Exoplanet spectroscopy infrared space telescope stellar atmospheric transit analysis.",
            "published_date": "2024-09-22",
            "access_tier": "Public",
            "source_channel": "Research_Portal",
            "is_verified": 9,  # Invalid binary flag (must be 0 or 1)
            "author_experience_years": 11.0,
            "citation_count": 64.0,
            "category": "Science",
        },
    ]

    # Add 5 rows with missing target value (`category` is NaN)
    missing_target_rows = [
        {
            "doc_id": f"DOC-910{idx}",
            "title": f"Unlabeled Archival Ingestion Document #{idx}",
            "content": (
                "Unclassified enterprise archive containing mixed technical, financial, and regulatory "
                "terminology awaiting automated category assignment."
            ),
            "published_date": f"2025-01-1{idx}",
            "access_tier": "Internal",
            "source_channel": "Enterprise_Wiki",
            "is_verified": 1,
            "author_experience_years": 6.0 + idx,
            "citation_count": 12.0 + idx * 3,
            "category": np.nan,
        }
        for idx in range(1, 6)
    ]

    df = pd.DataFrame(records + anomalous_rows + missing_target_rows)
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} document records at '{out_file}'.")
    return df


if __name__ == "__main__":
    generate_corpus()
