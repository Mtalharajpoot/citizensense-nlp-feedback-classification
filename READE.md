# CitizenSense: NLP-Based Public Feedback Classification and Insight System

AIRI Team PITB - AI Internship Task 4 (individual task)
Author: [MUHAMMAD TALHA FAROOQ]

CitizenSense reads a short piece of citizen feedback (English, Roman Urdu or mixed) and returns a structured insight card: category, sentiment, priority, keywords, a short summary, a suggested department and a suggested action. It is a **learning prototype**, not an official complaint-management system. Predictions are suggestions, not final decisions.

## Example

Input:

```
The road in our area has been broken for months and rainwater collects every day. Please fix it urgently.
```

Output:

```
Category:   Roads / Infrastructure
Sentiment:  Negative
Priority:   High
Keywords:   rainwater, road, rainwater collects, broken
Department: Municipal / Infrastructure (Works and Services)
Summary:    Citizen reports a Roads / Infrastructure problem (rainwater, road, rainwater collects, broken).
```

## Dataset

- **510 text samples**, 6 categories, **85 samples per category** (balanced): Roads / Infrastructure, Water Supply, Electricity, Sanitation, Health, Traffic.
- **Source: synthetic.** All samples are short feedback messages written for this project, drafted with AI assistance and reviewed and labeled by the author. They are not official government complaint data. Some samples are in Roman Urdu or mixed language.
- Labels per sample: category (main task), sentiment (Negative / Neutral / Positive) and priority (Low / Medium / High / Urgent).
- No private personal information (phone numbers, CNIC numbers, addresses) is included.
- Split: **train 357 / validation 76 / test 77** (about 70 / 15 / 15).
- Raw and processed data are kept separately in `data/raw/` and `data/processed/`.

### Preprocessing

Lowercasing where appropriate, removing extra spaces, duplicate samples, URLs and phone-number patterns, handling punctuation and missing values, tokenization (through the vectorizer / embedding model), label encoding and the train / validation / test split.

## Methodology

1. Create and label the dataset, then clean it.
2. **Baseline:** TF-IDF features with Logistic Regression (word n-grams). Character n-gram variants were also compared.
3. **Hugging Face / transformer part:** multilingual sentence embeddings ([MODEL NAME - copy from app/utils.py]) used in two ways: zero-shot category matching by embedding similarity, and as features for a Logistic Regression classifier (the final category model).
4. Sentiment and priority are predicted with separate trained classifiers.
5. **Insight generation:** keyword extraction, a short template-based summary, a suggested department and a suggested action.
6. Predictions with low confidence are flagged with `needs_manual_review`.
7. Demo through CLI and Streamlit.

## Results

Final category model: multilingual sentence embeddings + Logistic Regression, trained on train + validation (433 samples) and evaluated once on the test set (77 samples).

| Metric | Value |
|---|---|
| Accuracy | 0.792 |
| Macro F1 | 0.791 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Electricity | 0.71 | 0.77 | 0.74 |
| Health | 0.86 | 1.00 | 0.92 |
| Roads / Infrastructure | 0.67 | 0.62 | 0.64 |
| Sanitation | 0.86 | 0.92 | 0.89 |
| Traffic | 0.77 | 0.77 | 0.77 |
| Water Supply | 0.90 | 0.69 | 0.78 |

Other comparisons (5-fold cross-validation on train + validation unless noted):

| Method | Result |
|---|---|
| TF-IDF word + Logistic Regression (baseline) | CV accuracy 0.769 |
| TF-IDF character n-grams + Logistic Regression | CV accuracy 0.832 |
| Sentence embeddings + Logistic Regression | CV accuracy 0.855 |
| Zero-shot (embedding similarity), test set | accuracy 0.857, macro F1 0.862 |

The zero-shot result is only about 5 test samples better than the trained model on a 77-sample test set, so the two should be treated as comparable.

- **Best class:** Health (F1 0.92). **Weakest class:** Roads / Infrastructure (F1 0.64).
- **Most confused:** Roads / Infrastructure was predicted as Electricity 3 times (street-light ambiguity); Water Supply was predicted as Sanitation 2 times.
- Accuracy alone can hide weak classes; macro F1 gives every class equal weight.

## Error analysis

16 of 77 test samples were misclassified (see `outputs/error_analysis.csv`, with error type and a possible reason for each). Main patterns:

- **7 of 16 errors are Roman Urdu texts** (few Roman Urdu training examples).
- Street lights and similar topics overlap between Roads and Electricity.
- Very short texts and thank-you messages contain few category keywords.
- Some topics (for example challan / traffic fines) have few training samples.
- One pedestrian-bridge sample shows inconsistent labeling.

The "possible reason" column contains the author's hypotheses and has not been proven by further experiments.

Suggested improvements: clearer labeling guidelines, more Roman Urdu data, more samples for rare topics, human review for low-confidence predictions and multi-label classification for texts with more than one issue.

## Project structure

```
nlp_task_4/
├── data/            raw/, processed/, labels.csv
├── notebooks/       nlp_feedback_analysis.ipynb
├── app/             demo.py, streamlit_app.py, utils.py
├── models/          baseline_model.pkl, label_encoder.pkl, category / sentiment / priority models
├── outputs/         confusion_matrix.png, classification_report.txt,
│                    sample_predictions.csv, error_analysis.csv, ...
├── screenshots/     demo_input.png, demo_output.png, evaluation_results.png
├── report/          final_report.pdf
├── README.md
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# CLI demo
python app/demo.py

# Streamlit demo
streamlit run app/streamlit_app.py
```

The first run downloads the sentence-embedding model from Hugging Face, so an internet connection is needed.

## Sample predictions

`outputs/sample_predictions.csv` contains 30 example predictions with keywords, summary and suggested department. These are demo examples and are **not** used as the evaluation result; the evaluation numbers above come from the held-out test set.

## Limitations

- The dataset is small (510 samples) and synthetic, so the results do not show real-world performance.
- The test set has only 77 samples, so scores can move by several points with a few samples.
- Roman Urdu support is limited; most errors are in Roman Urdu texts.
- Each text gets one category, but real complaints can contain more than one issue.
- Summaries are template-based, and department names are suggestions only.
- Predictions should be reviewed by a person before any action is taken.

## Tools

Python, Google Colab, Pandas, NumPy, Scikit-learn, TF-IDF, Logistic Regression, Hugging Face Transformers / Sentence-Transformers, Matplotlib, Streamlit.

## Links

GitHub / Google Drive folder: [https://drive.google.com/drive/folders/1SvGAELE6yq1SxTvxbU1KrQCI04bFX9Ld?usp=sharing]