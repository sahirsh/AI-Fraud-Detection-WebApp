# Job Posting Fraud Detection API

Flask API that scores job listings for fraud risk using a TF-IDF + logistic regression model. A small Streamlit UI is included for trying requests against a running endpoint.

Fake job postings are rare in the wild, so the training set is skewed (~95% legitimate on the primary EMSCAD-style dataset). The model uses `class_weight="balanced"` and is evaluated with precision, recall, and F1 on the fraud class—not accuracy alone.

## Model

- **Features:** title, company profile, description, requirements, and benefits (cleaned text → TF-IDF unigrams/bigrams)
- **Classifier:** scikit-learn `LogisticRegression`
- **Training data:** `data/fake_job_postings.csv` (mixed labels) plus `data/Fake Postings.csv` (extra fraudulent examples, train-only, down-weighted)
- **Artifacts:** `ml/model.joblib`, `ml/vectorizer.joblib` (loaded at API startup; no retrain on deploy)

### Holdout metrics (primary dataset, threshold 0.5)

| Metric | Value |
|--------|------:|
| Fraud precision | 79.4% |
| Fraud recall | 84.4% |
| Fraud F1 | 0.82 |
| Accuracy | 98.2% |
| ROC-AUC | 0.98 |
| PR-AUC | 0.88 |

The API maps probability into risk bands in `api/inference.py` (very low / low / medium / high).

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run the API

From the project root (venv activated):

```bash
python -m api.app
```

Flask serves on `http://127.0.0.1:5000` by default.

- `GET /health` → `{"status": "ok"}`
- `POST /predict` with JSON:

```json
{
  "title": "...",
  "company_profile": "...",
  "description": "...",
  "requirements": "...",
  "benefits": "..."
}
```

Response includes `fraud_probability`, `label`, and `risk_level`.

## Streamlit UI

In a second terminal (venv activated):

```bash
streamlit run streamlit_app.py
```

The UI calls the API at `FRAUD_API_URL` (defaults to `http://127.0.0.1:5000`). Point it at your deployed API when needed:

```bash
# Windows PowerShell
$env:FRAUD_API_URL = "https://your-api.onrender.com"
streamlit run streamlit_app.py
```

Streamlit is a separate long-running Python app. It is **not** compatible with Vercel (Vercel expects serverless/static frontends). For a free hosted UI, use [Streamlit Community Cloud](https://streamlit.io/cloud) and set `FRAUD_API_URL` in the app secrets/env. Keep the Flask API on Render (or similar).

## Retrain

```bash
python ml/load_data.py
```

This rewrites `ml/model.joblib` and `ml/vectorizer.joblib`. Redeploy both files together.
