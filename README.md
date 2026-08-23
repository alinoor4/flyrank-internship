# FlyRank ML Internship — Capstone Research & Portfolio

> **Lane 1:** Ranking Signal Analysis & Opportunity Scoring (Action Queue Ranking)  
> **Author:** Ali Noor  
> **Deployed Research Paper:** [https://alinoor4.github.io/flyrank-internship/](https://alinoor4.github.io/flyrank-internship/)  
> **Dataset Credit:** Built on the [FlyRank](https://flyrank.ai/) Search Intelligence ML Internship dataset (~79M rows).

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Research%20Paper-0284c7?style=flat-square&logo=github)](https://alinoor4.github.io/flyrank-internship/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/capstone.ipynb?flush_cache=true)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## Executive Summary & Headline Results

This repository contains the end-to-end machine learning system, reproducible research notebooks, and published research paper for **Lane 1: Ranking Signal Analysis / Opportunity Scoring**. 

We addressed the enterprise editorial triage bottleneck: determining which URLs in a large publishing catalog (10,000+ articles) will yield the highest organic search return from an editorial refresh sprint (metadata rewrite, header restructuring, or content depth expansion) versus passive monitoring.

| Metric / Evaluation Split | Transparent Rule Baseline (Week 4) | Best ML Model: Gradient Boosting (depth=4) |
|---|---|---|
| **Validation Base Rate ($\alpha$)** | `0.1986` (19.86%) | `0.1986` (19.86%) |
| **Precision@10 (Holdout Domains)** | `0.9000` (90.0%) | **`1.0000` (100.0%)** |
| **Precision@20 (Holdout Domains)** | `0.8500` (85.0%) | **`1.0000` (100.0%)** |
| **Precision@50 (Holdout Domains)** | `0.8600` (86.0%) | **`1.0000` (100.0%)** (+14.0% lift) |
| **ROC-AUC (Holdout Domains)** | `0.9104` | **`0.9608`** |
| **PR-AUC (Holdout Domains)** | `0.7259` | **`0.8701`** |
| **Editorial Triage ROI** | 22.4 clicks/hr | **99.5 clicks/hr** ($9.3\times$ catalog leverage) |

---

## 3-Sentence Summary

- **What I Built:** An end-to-end machine learning content opportunity ranking system and interactive editorial triage playbook that predicts future organic click capture and assigns actionable refresh reason codes.
- **On What Data:** Evaluated across 10,000 mature content assets from 30 client domains using the FlyRank Search Intelligence warehouse, with strictly isolated 15-day pre/post observation windows and a zero-leakage client-grouped holdout split.
- **What It Showed:** Gradient Boosting achieved **100.0% Precision@50** and **0.9608 ROC-AUC** on unseen client domains (+14% over standard industry heuristics), demonstrating that targeted editorial triage captures **10.2% of search traffic in just 1.1% of review time (9.3× ROI lift)**.

---

## Research Progression & Weekly Assignment Notebooks

Every stage of the 8-week research track is documented in an executable, reproducible notebook in [`work/notebooks/`](work/notebooks/):

| Week | Milestone | Focus Area & Notebook | Colab Badge | Status |
|---|---|---|---|---|
| **Week 1** | ML-02 | Research Question Framing & Misallocation Cost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w01_research_question.ipynb?flush_cache=true) | ✅ Complete |
| **Week 2** | ML-03 | ML Task Framing & Prediction Target Definition (`clk_future >= 5`) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w02_ml_task_framing.ipynb?flush_cache=true) | ✅ Complete |
| **Week 3** | ML-04 | Data Contract, Schema Integrity & Quality Thresholds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w03_data_contract.ipynb?flush_cache=true) | ✅ Complete |
| **Week 3** | ML-05 | Feature Leakage Audit (Exclusion of `trend_pct`, `is_declining_label`) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w03_feature_leakage_check.ipynb?flush_cache=true) | ✅ Complete |
| **Week 4** | ML-06 | Signal Audit (Search impressions, striking rank, CTR) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w04_signal_audit.ipynb?flush_cache=true) | ✅ Complete |
| **Week 4** | ML-07 | Rule Baseline Score Formulation (Log impressions × striking multiplier) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w04_baseline_score.ipynb?flush_cache=true) | ✅ Complete |
| **Week 5** | ML-08 | Model Training on Client-Grouped Holdout Split (`GroupShuffleSplit`) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w05_model.ipynb?flush_cache=true) | ✅ Complete |
| **Week 6** | ML-09 | Validation Audit, Permutation Importance & Error Hand-Review | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w06_validation_audit.ipynb?flush_cache=true) | ✅ Complete |
| **Week 7** | ML-10 | Content Action Playbook (Dual-Engine Priority Queue, Reason Codes & ROI) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/w07_action_playbook.ipynb?flush_cache=true) | ✅ Complete |
| **Week 8** | ML-11 | **Capstone Notebook** (End-to-End Synthesis, Demo & Shareable Cuts) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alinoor4/flyrank-internship/blob/main/work/notebooks/capstone.ipynb?flush_cache=true) | ✅ Complete |

---

## Repository Structure

```text
├── index.html                  # Public deployed research paper (served via GitHub Pages)
├── submission/
│   └── paper_url.txt           # Deployment URL: https://alinoor4.github.io/flyrank-internship/
├── work/
│   ├── index.html              # Research paper mirror
│   ├── capstone_report.md      # Full 8-section research report matching internship rubric
│   ├── notebooks/              # All completed weekly assignment & capstone notebooks
│   │   ├── w01_research_question.ipynb
│   │   ├── w02_ml_task_framing.ipynb
│   │   ├── w03_data_contract.ipynb
│   │   ├── w03_feature_leakage_check.ipynb
│   │   ├── w04_signal_audit.ipynb
│   │   ├── w04_baseline_score.ipynb
│   │   ├── w05_model.ipynb
│   │   ├── w06_validation_audit.ipynb
│   │   ├── w07_action_playbook.ipynb
│   │   └── capstone.ipynb
│   ├── figures/                # Publication-ready charts (PNG & SVG)
│   │   ├── action_mix.png / .svg
│   │   ├── reason_codes.png / .svg
│   │   ├── cost_value_curve.png / .svg
│   │   └── model_vs_baseline_queue.png / .svg
│   └── outputs/                # Action queue exports and summary metrics
│       ├── ranked_action_queue.csv
│       └── w07_playbook_summary.json
├── scripts/                    # Reference pipeline utilities
├── docs/                       # Core dataset dictionaries and internship guides
├── requirements.txt            # Project dependencies
└── README.md
```

---

## Quickstart & Reproducibility

To reproduce the analysis and pipeline from a fresh clone:

```bash
# 1. Clone this repository
git clone https://github.com/alinoor4/flyrank-internship.git
cd flyrank-internship

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Execute the full Capstone notebook top-to-bottom
jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb --output work/notebooks/capstone.ipynb
```

---

## 5-Minute Showcase Demo Outline

- **The Decision & Problem** — Enterprise content teams manage 10,000+ published URLs but can only review 30–80 pages per month. Naive rules chase impressions and waste hours on zero-click SERPs (AI Overviews) while letting page-2 striking distance articles decay.
- **Data & Method** — 10,000 mature content assets across 30 client domains from FlyRank's warehouse release with strictly isolated 15-day pre/post observation windows and `GroupShuffleSplit` on `client_id` (0 domain overlap).
- **The Key Chart** — Editorial Cost-Value Curve (**Figure 3**) showing how the prioritized queue achieves extreme editorial leverage ($9.3\times$ ROI lift).
- **One Honest Result & Error Audit** — Gradient Boosting achieves **$100.0\%$ Precision@50** vs Rule Baseline's **$86.0\%$** (ROC-AUC = 0.9608); error audit on broad queries cannibalized by Google AI Overviews.
- **The Recommendation & Playbook** — Dual-Engine Priority Queue, automated reason codes (`STRIKING_DISTANCE_HIGH_OPS`, `LOW_CTR_OPPORTUNITY`), 5-point checklist, and strict No-Go policies.

---

## Public Safety & Data Credit

- **No Private Data**: All client names, domain URLs, credentials, and raw query strings have been stripped or pseudonymized.
- **Honest Claim Language**: All findings report measured associations in the FlyRank 2026 search dataset; no causal claims are made regarding Google's internal ranking weights without an experimental intervention design.
- **Attribution**: Built on the [FlyRank](https://flyrank.ai/) ML Internship dataset.
