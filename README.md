# Fake News Classification and Explainability

Summer 2026 MPCS self-driven project (University of Chicago). Mentor: Dhairya Karna.


## Project Overview

This project has two parts:

**Part 1 — Dataset Leakage Investigation (ISOT).** Applying SHAP and LIME to
DistilBERT/RoBERTa classifiers trained on the ISOT Fake News dataset revealed that
near-perfect accuracy (99.97%+) was driven substantially by a wire-service dateline
artifact and a near-perfect subject-category/label correlation, rather than genuine
content understanding. Controlled experiments stripping both shortcuts are documented
in notebooks 02-06.

**Part 2 — LLM-Generated Explanations (LIAR).** A pipeline that converts SHAP
attributions into natural-language explanations using Google Gemini, evaluated on the
LIAR dataset (which does not exhibit ISOT's leakage issues). Includes a three-metric
faithfulness evaluation and a 12-participant user study. Notebooks 07-10.

## Repository Structure

```
notebooks/     Jupyter notebooks, numbered in project order (see below)
app/           Streamlit dashboard (live prediction + explanation demo)
report/        Final written report
docs/          User study materials and results
```

### Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `notebooks/isot/01_fake_news_model_256.ipynb` | ISOT EDA, train/val/test split, TF-IDF baseline, DistilBERT + RoBERTa fine-tuning (256 tokens) |
| 02 | `notebooks/isot/02_fake_news_model_512.ipynb` | Sequence-length experiment (512 vs 256 tokens) |
| 03 | `notebooks/isot/03_shap_lime_analysis_isot.ipynb` | SHAP/LIME, disagreement analysis, Reuters dateline discovery |
| 04 | `notebooks/isot/04_retrain_isot.ipynb` | Retraining from scratch on dateline-stripped ISOT |
| 05 | `notebooks/isot/05_subject_check_isot.ipynb` | Subject-category leakage quantification |
| 06 | `notebooks/isot/06_subject_holdout_isot.ipynb` | Subject-holdout generalization test |
| 07 | `notebooks/isot/07_combined_leakage_isot.ipynb` | Combined dateline-stripped + subject-holdout controlled test |
| 08 | `notebooks/liar/08_liar_fake_news.ipynb` | Baseline/DistilBERT/RoBERTa trained on LIAR |
| 09 | `notebooks/liar/09_liar_shap_lime_pipeline.ipynb` | SHAP and LIME on LIAR |
| 10 | `notebooks/liar/10_liar_full_llm_pipeline.ipynb` | Full SHAP + LIME + Gemini LLM explanation pipeline |
| 11 | `notebooks/liar/11_liar_faithfulness_evaluation.ipynb` | Faithfulness evaluation (token coverage, directional accuracy, semantic consistency) |
| 12 | `notebooks/liar/12_generate_user_study_examples.ipynb` | Generates the 10 example cases used in the user study |

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # get a free key at https://aistudio.google.com/apikey
```

Datasets are downloaded automatically within the notebooks (ISOT via `kagglehub`, LIAR
via Hugging Face `datasets`) — no manual download needed.

Trained models are not included in this repository (too large for git). Re-run
notebooks 01 and 08 to regenerate them, or see the notebooks for expected save paths.

## Running the Dashboard

```bash
cd app
pip install -r requirements_dashboard.txt
export GEMINI_API_KEY="your-key-here"
streamlit run app.py
```

Requires the LIAR-trained DistilBERT model (from notebook 08) to be present at
`./distilbert_liar_final` relative to `app.py`.

## Key Results

**ISOT leakage investigation:** dateline-stripping + subject-holdout, combined,
dropped DistilBERT accuracy from 99.98% to 21.91% (below a trivial majority-class
baseline), demonstrating severe reliance on dataset artifacts.

**LIAR faithfulness evaluation:** 78% token coverage, 100% directional accuracy
(on checkable cases), 0.88 semantic consistency across repeated LLM generations.

**User study (n=12):** 3.90/5 average clarity, 3.62/5 average trust, 83.3% preferred
plain-English explanations over raw SHAP output.

Full details, tables, and discussion in [`report/Fake News Classification and Explainability.pdf`]
## References

See the References section of the final report for full citations (Geirhos et al. 2020,
Ribeiro et al. 2016, Lundberg & Lee 2017, and others).
