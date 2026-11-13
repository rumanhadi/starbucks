import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Predictor App", page_icon="🚀", layout="centered")

@st.cache_resource
def load_artifact():
    return joblib.load("model.joblib")

artifact = load_artifact()
pipeline = artifact["pipeline"]
num_cols = artifact["num_cols"]
cat_cols = artifact["cat_cols"]
num_defaults = artifact["num_defaults"]
cat_options = artifact["cat_options"]
target_name = artifact["target_name"]

st.title(f"🚀 {target_name} Prediction App")
st.write("Adjust the input features below and click **Predict**.")

user_inputs = {}

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(cat_cols):
        container = col1 if i % 2 == 0 else col2
        user_inputs[col] = container.selectbox(col, options=cat_options[col])
        
    for i, col in enumerate(num_cols):
        container = col1 if (i + len(cat_cols)) % 2 == 0 else col2
        default_val = float(num_defaults.get(col, 0.0))
        user_inputs[col] = container.number_input(col, value=default_val)
        
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([user_inputs])
    prediction = pipeline.predict(input_df)[0]
    st.success(f"### Predicted {target_name}: **{prediction:.2f}**")
