# Capstone Report — Lane 1: Ranking Signal Analysis & Action Queue Ranking

- **Author:** Ali Noor
- **Lane:** Lane 1 — Ranking Signal Analysis / Opportunity Scoring (Action Queue Ranking)
- **Repo:** [https://github.com/alinoor4/flyrank-internship](https://github.com/alinoor4/flyrank-internship)
- **Date:** August 2026
- **Deployed Research Paper:** [https://alinoor4.github.io/flyrank-internship/](https://alinoor4.github.io/flyrank-internship/)

---

## 1. Problem Framing

### Decision Context
In enterprise organic search optimization, editorial teams manage large content libraries (thousands to tens of thousands of published articles) under finite bandwidth constraints. Content teams cannot manually review or rewrite every page each month. The operational problem is **editorial triage**: determining which existing content assets to prioritize for metadata rewrites, header restructurings, or content depth expansions in upcoming editorial sprints.

### Unit of Analysis & Target Definition
- **Unit of Analysis**: An individual pseudonymized content item (`content_id`) associated with a client domain (`client_id`).
- **Output Artifact**: A continuous **Dual-Engine Priority Score** ($0\text{--}100$) and an operational **Ranked Action Queue** with categorical **Reason Codes** (`STRIKING_DISTANCE_HIGH_OPS`, `LOW_CTR_OPPORTUNITY`, `THIN_CONTENT_HIGH_IMP`, `DECAY_REFRESH_CANDIDATE`, `MONITOR_ONLY`) and human review effort estimates.
- **Target Proxy**: Binary high future search performance indicator ($Y = 1$ if `clk_future >= 5` clicks in the 15-day forward target window; base rate $\alpha = 0.2473$ on the full cohort and $\alpha_{\text{val}} = 0.1986$ on out-of-fold validation domains).

### Operational Tradeoffs & Cost of Misallocation
- **Cost of False Positives**: Assigning high priority to pages that capture zero future clicks (e.g., zero-click informational SERP queries or algorithmically depressed topics) wastes $1.5\text{--}3.0$ editorial hours per URL ($\$75\text{--}\$150$ in labor cost).
- **Cost of False Negatives**: Failing to flag decaying ranking assets in striking positions ($4\text{--}30$) forfeits high-intent organic search volume, allowing competing domains to capture market share.
- **Why Machine Learning Helps**: Machine learning ensembles model non-linear interactions between historical impression volume, ranking position, click-through efficiency, and content length, achieving $\text{Precision@50} = 1.000$ (100% precision) compared to $0.860$ for rule-based heuristics on unseen client domains.

---

## 2. Data Safety & Leakage Prevention

### Data Sources & Temporal Window Isolation
- **Dataset**: FlyRank Search Intelligence Warehouse release (`content_refresh_anonymized.csv` and Hugging Face DuckDB warehouse release `hf://datasets/FlyRank/internship-warehouse`).
- **Feature Window**: Days 1–15 of March 2026 (`2026-03-01` to `2026-03-15`), extracting historical impressions (`imp_prev30`), clicks (`clk_prev30`), average ranking position (`pos_prev30`), CTR (`ctr_prev30`), content word count, and query visibility.
- **Target Window**: Days 16–31 of March 2026 (`2026-03-16` to `2026-03-31`), extracting forward clicks (`clk_future`).

### Strict Exclusions & Public Safety
- **Leakage-Derived Columns Excluded**: `trend_direction`, `trend_pct`, and `is_declining_label` are strictly excluded from the feature space as they are computed using target-window metrics.
- **Pseudonymous Entity Safety**: `client_id` and `content_id` are used strictly for grouping and joining—never as feature inputs—preventing tree models from memorizing specific client scale or domain authority.
- **Public Hygiene**: Zero client names, raw domain URLs, private query strings, or credentials appear anywhere in the codebase or public artifacts.

### Missingness Handling
Following the `flyrank-data` contract, missing values are handled via structural indicator flags (`has_word_count`, `has_visible_queries`) and zero-filling the cleaned numeric columns, preventing artificial categorical leakage.

---

## 3. Baseline Formulation

### Transparent Heuristic Formulation
To benchmark machine learning models against a transparent operational standard, we formulated the **Week 4 Rule-Based Baseline Score**:

$$\text{Baseline Score} = \log(1 + \text{imp\_prev30}) \times \text{striking\_mult} \times \text{ctr\_gap\_mult}$$

where:
- $\text{striking\_mult} = 1.5$ if $3.0 < \text{pos\_prev30} \le 30.0$, else $1.0$.
- $\text{ctr\_gap\_mult} = 1.3$ if $\text{ctr\_prev30} < 1.0\%$, else $1.0$.

### Baseline Validation Performance
On the out-of-fold client validation split (8 unseen client domains, 2,019 content items, base rate $0.1986$):
- **Precision@10**: $0.9000$ (90.0%)
- **Precision@20**: $0.8500$ (85.0%)
- **Precision@50**: $0.8600$ (86.0%)
- **ROC-AUC**: $0.9104$
- **PR-AUC**: $0.7259$

The baseline provides a competitive heuristic but exhibits false-positive leakage when high impressions occur on deeply buried pages ($> 30$) or when CTR deficits stem from zero-click SERP layouts.

---

## 4. Model & Feature Architecture

### Candidate Architectures
1. **Logistic Regression (L2 Regularization, $C=1.0$)**: Calibrated linear baseline for inspectable coefficients.
2. **Decision Tree Classifier (`max_depth=4`)**: Interpretable non-linear threshold tree.
3. **Random Forest Classifier (`n_estimators=100`, `max_depth=6`)**: Bagged ensemble capturing feature interactions with variance reduction.
4. **Gradient Boosting Classifier (`n_estimators=100`, `learning_rate=0.05`, `max_depth=4`)**: Sequential residual minimization optimizing non-linear ranking boundaries.

### Feature Specification
- **Search Demand & Visibility**: `imp_prev30` (historical impressions), `clk_prev30` (historical clicks), `pos_prev30_clean` (average ranking position, nulls imputed to $99.0$), `ctr_prev30` (historical click-through rate).
- **Content Metadata**: `word_count_clean`, `has_word_count`, `visible_queries_clean`, `has_visible_queries`.
- **Categorical Formats**: One-hot encoded `content_type` (`type_keyword article`, `type_feedly article`, etc.).

---

## 5. Evaluation & Validation Design

### Validation Design: Client-Grouped Split
To rigorously test generalization to unseen client websites, we enforce a **Client-Grouped Out-of-Fold Split** (`GroupShuffleSplit` on `client_id` with 25% holdout):
- **Training Cohort**: 7,981 content items across 22 client domains (Base Rate: $0.2596$).
- **Validation Cohort**: 2,019 content items across 8 distinct client domains (Base Rate: $0.1986$).
- **Client Domain Overlap**: Strictly $0$ ($S_{\text{train}} \cap S_{\text{val}} = \emptyset$).

### Comprehensive Performance Comparison Table

| Model Architecture | Validation Base Rate | Precision@10 | Precision@20 | Precision@50 | ROC-AUC | PR-AUC | Lift vs Base Rate (P@50) |
|---|---|---|---|---|---|---|---|
| **Rule Baseline (Week 4)** | 0.1986 (19.9%) | 0.9000 (90.0%) | 0.8500 (85.0%) | 0.8600 (86.0%) | 0.9104 | 0.7259 | $4.33\times$ |
| **Logistic Regression** | 0.1986 (19.9%) | 1.0000 (100.0%) | 1.0000 (100.0%) | 1.0000 (100.0%) | 0.9487 | 0.8519 | $5.04\times$ |
| **Decision Tree (`depth=4`)** | 0.1986 (19.9%) | 0.8000 (80.0%) | 0.8500 (85.0%) | 0.8800 (88.0%) | 0.9563 | 0.8321 | $4.43\times$ |
| **Random Forest (`depth=6`)** | 0.1986 (19.9%) | 1.0000 (100.0%) | 1.0000 (100.0%) | 0.9400 (94.0%) | 0.9592 | 0.8639 | $4.73\times$ |
| **Gradient Boosting (`depth=4`)** | **0.1986 (19.9%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **0.9608** | **0.8701** | **$5.04\times$** |

### Key Validation Findings
- **Gradient Boosting** achieved top discrimination performance ($\text{ROC-AUC} = 0.9608$, $\text{PR-AUC} = 0.8701$) and flawless top-tier triage ($\text{Precision@50} = 1.0000$), delivering a $+14.0\%$ precision improvement over the Rule Baseline ($0.8600 \to 1.0000$).
- **PR-AUC Lift**: Gradient Boosting improved PR-AUC from $0.7259 \to 0.8701$ ($+19.9\%$ relative lift), confirming robust ranking across all probability thresholds.

---

## 6. Interpretation & Error Analysis

### Feature & Permutation Importance

| Feature | Gini Importance (Tree Split) | Permutation Importance (Mean ROC-AUC Drop) | Permutation Importance Std |
|---|---|---|---|
| `clk_prev30` | 0.9141 | 0.0541 | 0.0029 |
| `imp_prev30` | 0.0465 | 0.1245 | 0.0047 |
| `ctr_prev30` | 0.0171 | 0.0030 | 0.0010 |
| `pos_prev30_clean` | 0.0139 | 0.0143 | 0.0025 |
| `word_count_clean` | 0.0079 | 0.0004 | 0.0007 |
| `type_keyword article` | 0.0002 | 0.0000 | 0.0000 |
| `has_word_count` | 0.0002 | 0.0000 | 0.0000 |

- **Primary Predictive Driver**: While `clk_prev30` dominates split frequency, `imp_prev30` produces the largest drop in out-of-fold ROC-AUC ($0.1245$) when permuted, confirming that historical search demand volume is the foundation of future traffic potential.
- **Secondary Refinements**: `pos_prev30_clean` and `ctr_prev30` act as non-linear filters, differentiating striking-distance opportunities from buried pages.

### Hand-Audited Concrete Error Modes
1. **False Positive (Case 1 — `content_e32ec994d05a`)**:
   - *Features*: $\text{imp} = 5,363$, $\text{pos} = 4.2$, $\text{ctr} = 0.43\%$, $\text{word\_count} = 2,740$.
   - *Model Confidence*: $P(Y=1) = 0.9760$ | *Actual Clicks*: $1$ ($Y=0$).
   - *Root Cause*: High impressions on broad definitional search queries dominated by Google AI Overviews and featured answer boxes, resulting in zero-click user behavior despite top-5 ranking.
2. **False Positive (Case 2 — `content_291e5716005f`)**:
   - *Features*: $\text{imp} = 9,774$, $\text{pos} = 15.7$, $\text{ctr} = 0.19\%$, $\text{word\_count} = 2,442$.
   - *Model Confidence*: $P(Y=1) = 0.9672$ | *Actual Clicks*: $3$ ($Y=0$).
   - *Root Cause*: High second-page impression volume with low click efficiency narrowly missing the 5-click binary threshold.
3. **False Negative (Case 3 — `content_dbaec841d4cf`)**:
   - *Features*: $\text{imp} = 266$, $\text{pos} = 8.2$, $\text{ctr} = 0.00\%$, $\text{word\_count} = 2,902$.
   - *Model Confidence*: $P(Y=1) = 0.0100$ | *Actual Clicks*: $5$ ($Y=1$).
   - *Root Cause*: Low historical volume in the feature window that experienced sudden organic interest expansion during the target period.

---

## 7. Recommendations & Content Action Playbook

### Dual-Engine Triage System
To integrate machine learning confidence with operational SEO opportunity, we deploy a **Dual-Engine Priority Score**:

$$\text{Priority Score} = 0.60 \times (P(\text{High Performer}) \times 100) + 0.40 \times \text{Baseline Score}_{\text{norm}}$$

### Action Taxonomy & Reason Codes

| Archetype / Trigger | Reason Code | Recommended Action | Est. Review Time | Value Tier |
|---|---|---|---|---|
| Rank 4–30, Imp $\ge 100$, Prob $\ge 0.50$ | `STRIKING_DISTANCE_HIGH_OPS` | `REFRESH_METADATA_AND_HEADERS` | 1.5 hrs | HIGH |
| Rank $\le 10$, CTR $< 1.0\%$, Imp $\ge 100$ | `LOW_CTR_OPPORTUNITY` | `REWRITE_META_DESCRIPTION_AND_TITLE` | 1.0 hrs | HIGH |
| Imp $\ge 500$, Word Count $< 1,000$ | `THIN_CONTENT_HIGH_IMP` | `EXPAND_CONTENT_DEPTH` | 3.0 hrs | MEDIUM |
| Imp $\ge 200$, Prob $\ge 0.60$ | `DECAY_REFRESH_CANDIDATE` | `REFRESH_AND_UPDATE_FRESHNESS` | 2.0 hrs | HIGH |
| Deep rank ($> 30$) or Imp $< 100$ | `MONITOR_ONLY` | `MONITOR` | 0.2 hrs | LOW |

### Editorial ROI & Cost-Value Tiers

| Queue Tier | Content Count | Editorial Review Time | Forward Clicks Captured | Share of Catalog Clicks | Precision@K | Editorial ROI (Clicks/Hr) |
|---|---|---|---|---|---|---|
| **Top 10 Priority** | 10 | 15.0 hrs | 1,974 | 2.7% | 100.0% | **131.6** |
| **Top 20 Priority** | 20 | 30.0 hrs | 3,280 | 4.5% | 100.0% | **109.3** |
| **Top 50 Priority** | 50 | 75.0 hrs | 7,460 | 10.2% | 100.0% | **99.5** |
| **Top 100 Priority** | 100 | 150.0 hrs | 11,513 | 15.8% | 100.0% | **76.8** |
| **Top 500 Priority** | 500 | 750.0 hrs | 28,687 | 39.3% | 99.4% | **38.2** |
| **Entire Catalog** | 10,000 | 6,815.3 hrs | 73,036 | 100.0% | 24.7% | **10.7** |

*Key Insight*: Editorial teams executing on the **Top 50 Prioritized URLs** capture **$10.2\%$ of total portfolio search traffic** in just **$1.1\%$ of total catalog review time**, generating **$99.5\text{ clicks/hour}$** ($9.3\times$ higher leverage than unprioritized reviews).

### Human Review Checklist & No-Go Policies
- **Mandatory Review Checklist**:
  1. *Search Intent Alignment*: Verify that query intent matches page value proposition.
  2. *SERP Layout Audit*: Inspect whether Google AI Overviews or featured snippets suppress organic CTR.
  3. *Factual Verification*: Check statistics, pricing, and compliance requirements.
  4. *Cannibalization Prevention*: Confirm no competing internal URL targets the same query cluster.
  5. *Technical Health*: Validate page loading speed, schema markup, and response codes.
- **Strict No-Go Policies**: Never automate direct CMS publishing; require manual sign-off for YMYL (medical, legal, financial) content and core brand pages.

---

## 8. Reproducibility & Environment

### Environment & Execution
- **Python Version**: Python 3.10+ / 3.14
- **Core Dependencies**: `duckdb>=1.1.0`, `pandas>=2.2.0`, `numpy>=1.26.0`, `scikit-learn>=1.4.0`, `matplotlib>=3.8.0`
- **Random Seed**: `random_state = 42` enforced across all dataset splits and estimators.

### Commands to Reproduce from Fresh Clone
```bash
# 1. Clone repository
git clone https://github.com/alinoor4/flyrank-internship.git
cd flyrank-internship

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt

# 4. Run full Capstone notebook top to bottom
jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb --output work/notebooks/capstone.ipynb
```

---

## 9. 5-Minute Showcase Demo Outline

1. **Minute 1: The Decision & Problem**
   - *The Reality:* Enterprise content teams manage 10,000+ published URLs but can only review 30–80 pages per month.
   - *The Trap:* Naive rules chase impressions and waste hours on zero-click SERPs (AI Overviews) while letting page-2 striking distance articles decay.
2. **Minute 2: Data & Method**
   - *Dataset:* 10,000 mature content assets across 30 client domains from FlyRank's warehouse release.
   - *Hygiene:* Strictly isolated 15-day pre/post observation windows; out-of-domain evaluation using `GroupShuffleSplit` on `client_id` (0 domain overlap).
3. **Minute 3: The Key Chart (Editorial Cost-Value Curve)**
   - *Figure 3 Walkthrough:* Showing how the prioritized queue achieves extreme editorial leverage, capturing 10.2% of traffic in 1.1% of review time and 39.3% of traffic in 11.0% of review time.
4. **Minute 4: One Honest Result & Error Audit**
   - *Metric:* Gradient Boosting achieved **100.0% Precision@50** vs the Rule Baseline's **86.0%** (against a 19.86% holdout base rate, ROC-AUC = 0.9608).
   - *Error Mode:* Hand-audited false positives show that broad definitional queries get cannibalized by Google AI answers—proving why decision-support triage beats full automation.
5. **Minute 5: The Recommendation & Playbook**
   - *Dual-Engine Queue:* Automated Reason Codes (`STRIKING_DISTANCE_HIGH_OPS`, `LOW_CTR_OPPORTUNITY`, etc.) combined with a 5-Point Editorial Review Checklist and strict No-Go policies.

---

## Claims Checklist & Acknowledgments

- **Observed / Measured Framing**: All claims reflect measured associations in the FlyRank 2026 search dataset; no causal claims are made without an experimental intervention design.
- **No Algorithm Claims**: Findings describe portfolio traffic patterns, not Google's proprietary search ranking weights.
- **Base Rate Transparency**: Base rates ($\alpha = 24.73\%$ overall, $\alpha_{\text{val}} = 19.86\%$ out-of-fold) are reported alongside all Precision@K and ROC-AUC metrics.
- **Data Credit**: Built on the [FlyRank](https://flyrank.ai/) ML Internship dataset.

