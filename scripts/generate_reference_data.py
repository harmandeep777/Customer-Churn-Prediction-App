import os
import json
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("blastchar/telco-customer-churn")
path = os.path.join(path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(path)

df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

reference = {'numeric': {}, 'categorical': {}}

for col in numeric_cols:
    reference['numeric'][col] = {
        'min': float(df[col].min()),
        'max': float(df[col].max()),
        'median': float(df[col].median())
    }

for col in categorical_cols:
    reference['categorical'][col] = sorted(df[col].dropna().unique().tolist())

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'reference_data.json')
with open(out_path, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'Wrote exact reference data to {out_path}')
