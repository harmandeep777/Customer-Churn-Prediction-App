import os
import sys
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
import transformer

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
REFERENCE_PATH = os.path.join(BASE_DIR, 'app', 'reference_data.json')

st.set_page_config(page_title='Customer Churn Predictor', page_icon='H', layout='wide')

st.markdown("""
<style>
.main-header {
    padding: 1.5rem 2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #636EFA 0%, #4c56d6 100%);
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    margin: 0;
    font-size: 1.9rem;
}
.main-header p {
    margin: 0.3rem 0 0 0;
    opacity: 0.9;
    font-size: 0.95rem;
}
.result-card {
    padding: 1.6rem;
    border-radius: 14px;
    text-align: center;
    margin-top: 1rem;
}
.result-card-safe {
    background: #e8f8f0;
    border: 1px solid #22c55e;
}
.result-card-risk {
    background: #fdecea;
    border: 1px solid #ef4444;
}
.result-card h2 {
    margin: 0;
    font-size: 1.6rem;
}
.result-card p {
    margin: 0.4rem 0 0 0;
    color: #555;
}
div[data-testid="stMetric"] {
    background: #f8f9fb;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_reference_data():
    with open(REFERENCE_PATH) as f:
        return json.load(f)

model = load_model(MODEL_PATH)
reference = load_reference_data()
numeric_ref = reference['numeric']
categorical_ref = reference['categorical']

st.markdown("""
<div class="main-header">
    <h1> Customer Churn Predictor</h1>
    <p>Predict whether a telecom customer is likely to churn, one customer at a time or in bulk.</p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs([' Single Prediction', ' Batch Upload'])

st.sidebar.title('Customer Form')
st.sidebar.caption('Fill this in for the Single Prediction tab')

tenure = st.sidebar.slider('tenure', int(numeric_ref['tenure']['min']), int(numeric_ref['tenure']['max']), int(numeric_ref['tenure']['median']))
MonthlyCharges = st.sidebar.slider('MonthlyCharges', float(numeric_ref['MonthlyCharges']['min']), float(numeric_ref['MonthlyCharges']['max']), float(numeric_ref['MonthlyCharges']['median']))
TotalCharges = st.sidebar.slider('TotalCharges', float(numeric_ref['TotalCharges']['min']), float(numeric_ref['TotalCharges']['max']), float(numeric_ref['TotalCharges']['median']))
gender = st.sidebar.selectbox('gender', categorical_ref['gender'])
Partner = st.sidebar.selectbox('Partner', categorical_ref['Partner'])
Dependents = st.sidebar.selectbox('Dependents', categorical_ref['Dependents'])
PhoneService = st.sidebar.selectbox('PhoneService', categorical_ref['PhoneService'])
MultipleLines = st.sidebar.selectbox('MultipleLines', categorical_ref['MultipleLines'])
InternetService = st.sidebar.selectbox('InternetService', categorical_ref['InternetService'])
OnlineSecurity = st.sidebar.selectbox('OnlineSecurity', categorical_ref['OnlineSecurity'])
OnlineBackup = st.sidebar.selectbox('OnlineBackup', categorical_ref['OnlineBackup'])
DeviceProtection = st.sidebar.selectbox('DeviceProtection', categorical_ref['DeviceProtection'])
TechSupport = st.sidebar.selectbox('TechSupport', categorical_ref['TechSupport'])
StreamingTV = st.sidebar.selectbox('StreamingTV', categorical_ref['StreamingTV'])
StreamingMovies = st.sidebar.selectbox('StreamingMovies', categorical_ref['StreamingMovies'])
Contract = st.sidebar.selectbox('Contract', categorical_ref['Contract'])
PaperlessBilling = st.sidebar.selectbox('PaperlessBilling', categorical_ref['PaperlessBilling'])
PaymentMethod = st.sidebar.selectbox('PaymentMethod', categorical_ref['PaymentMethod'])

user_data = {
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'gender': gender,
    'Partner': Partner,
    'Dependents': Dependents,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod
}

Col = pd.DataFrame([user_data])

cat_col = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

with tab_single:
    pred = model.predict(Col)
    proba = model.predict_proba(Col)[0][1]

    left, right = st.columns([1, 1.3])

    with left:
        if pred[0] == 0:
            st.markdown(f"""
            <div class="result-card result-card-safe">
                <h2> Likely to Stay</h2>
                <p>Estimated churn probability: <b>{proba:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-card-risk">
                <h2> At Risk of Churning</h2>
                <p>Estimated churn probability: <b>{proba:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric('Tenure', f'{tenure} mo')
        c2.metric('Monthly', f'${MonthlyCharges:,.0f}')
        c3.metric('Contract', Contract)

    with right:
        gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=proba * 100,
            number={'suffix': '%'},
            title={'text': 'Churn Risk'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#ef4444' if proba >= 0.5 else '#22c55e'},
                'steps': [
                    {'range': [0, 40], 'color': '#e8f8f0'},
                    {'range': [40, 70], 'color': '#fff7e6'},
                    {'range': [70, 100], 'color': '#fdecea'}
                ]
            }
        ))
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)

with tab_batch:
    uploaded_file = st.file_uploader('Choose a csv file', type='csv')

    if uploaded_file:
        try:
            upload_df = pd.read_csv(uploaded_file)
            X = upload_df.drop(columns=['Churn'], errors='ignore')
            raw_preds = model.predict(X)
        except Exception as e:
            st.error(f'Could not generate predictions from this file: {e}')
            st.stop()

        total = len(raw_preds)
        churned = int(sum(str(p) in ['1', '1.0'] for p in raw_preds))
        m1, m2, m3 = st.columns(3)
        m1.metric('Total Customers', f'{total:,}')
        m2.metric('Predicted to Churn', f'{churned:,}')
        m3.metric('Churn Rate', f'{churned / total:.1%}' if total else '0%')

        predicts = pd.DataFrame({'Prediction': raw_preds})
        with st.expander('View raw predictions table'):
            st.dataframe(predicts, use_container_width=True)

        class_counts = predicts['Prediction'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        class_counts['Class'] = class_counts['Class'].map({0: 'Class 0', 1: 'Class 1', '0': 'Class 0', '1': 'Class 1'})

        fig = px.pie(
            class_counts,
            values='Count',
            names='Class',
            title='Ratio of Predicted Churn (0s and 1s)',
            color='Class',
            color_discrete_map={'Class 0': '#22c55e', 'Class 1': '#ef4444'},
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

        if 'gender' in upload_df.columns:
            gen = upload_df['gender'].value_counts().reset_index()
            gen.columns = ['gender', 'Count']

            gen_fig = px.pie(
                gen,
                values='Count',
                names='gender',
                title='Gender Distribution in Uploaded Dataset',
                hole=0.45
            )
            st.plotly_chart(gen_fig, use_container_width=True)

        upload_df['Churn'] = ['Class 1' if str(p) in ['1', '1.0'] else 'Class 0' for p in raw_preds]

        valid_cat_cols = [col for col in cat_col if col in upload_df.columns]

        for col in valid_cat_cols:
            fig = px.histogram(
                upload_df,
                x=col,
                color='Churn',
                barmode='group',
                color_discrete_sequence=['#22c55e', '#ef4444'],
                title=f'Count Plot of {col} by Predicted Churn'
            )
            st.plotly_chart(fig, use_container_width=True)

        if all(c in upload_df.columns for c in ['tenure', 'MonthlyCharges', 'TotalCharges']):
            fig = px.scatter_matrix(
                upload_df,
                dimensions=['tenure', 'MonthlyCharges', 'TotalCharges'],
                color='Churn',
                color_discrete_map={'Class 0': '#22c55e', 'Class 1': '#ef4444'},
                title='Pair Plot / Scatter Matrix'
            )
            fig.update_traces(diagonal_visible=True)
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Upload a CSV with the same columns as the Telco Churn dataset to get bulk predictions and charts.')
