# Customer Churn Predictor

A Streamlit app that predicts whether a telecom customer will churn, trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Supports a single customer form in the sidebar or a bulk CSV upload with visual breakdowns of predictions.

## Project structure

```
customer-churn-predictor/
├── app/
│   ├── app.py               Streamlit entry point
│   └── reference_data.json  Slider bounds and dropdown options for the sidebar form
├── src/
│   ├── __init__.py
│   └── transformer.py       Custom sklearn transformers used inside the trained pipeline
├── models/
│   └── churn_model.pkl      Trained XGBoost pipeline (imblearn Pipeline)
├── notebooks/
│   └── ChurnProject.ipynb   Training notebook: EDA, preprocessing, model search
├── scripts/
│   └── generate_reference_data.py   One-time local script to regenerate reference_data.json from the real dataset
├── .streamlit/
│   └── config.toml          App theme
├── requirements.txt         Runtime dependencies (what the deployed app needs)
├── requirements-dev.txt     Adds notebook/EDA/kagglehub deps on top of requirements.txt
└── .gitignore
```

## Model

`models/churn_model.pkl` is an `imblearn.pipeline.Pipeline` with these steps:

1. `TotalChargesTransformer` — drops `customerID`, coerces `TotalCharges` to numeric
2. `FeatureEngineeringTransformer` — buckets `tenure` into short/mid/long-term customer
3. `ColumnTransformer` — median imputation + scaling on numeric columns, most-frequent imputation + one-hot encoding on categorical columns
4. `RandomOverSampler` — balances the churn classes during training only
5. `XGBClassifier`

Because the pipeline was pickled with `transformer.py` imported as a top-level module, `app.py` adds `src/` to `sys.path` before importing it, so the pickle can resolve `TotalChargesTransformer` and `FeatureEngineeringTransformer` correctly. Don't move or rename `src/transformer.py` without re-pickling the model.

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
streamlit run app/app.py
```

The app reads `app/reference_data.json` for the sidebar's slider ranges and dropdown options, no Kaggle account or network access needed at runtime.

The categorical options and numeric medians in `reference_data.json` are exact, pulled directly from the trained pipeline's fitted `OneHotEncoder` and `SimpleImputer`. The slider min/max bounds are conservative placeholders. To replace them with the dataset's exact min/max, run once locally:

```bash
pip install kagglehub
python scripts/generate_reference_data.py
```

This overwrites `app/reference_data.json` with exact values and requires Kaggle API credentials locally (`~/.kaggle/kaggle.json`), but only for this one-time regeneration, not for running or deploying the app.

## Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo.
3. Set the main file path to `app/app.py`.
4. Deploy. No secrets are required, the app has no external network dependency at runtime.

## Retraining

Open `notebooks/ChurnProject.ipynb` with the dev dependencies installed (`pip install -r requirements-dev.txt`). Re-saving `churn_model.pkl` after changes to `src/transformer.py` is required, since the pickle is tied to the transformer class definitions at save time.
