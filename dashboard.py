import streamlit as st
import pandas as pd
import os
from config import *

st.set_page_config(layout="wide", page_title="Garbage Classification Dashboard")

# --- CSS Injection to fix Image Cropping instantly without a re-run ---
st.markdown("""
    <style>
    /* Add vertical breathing room and prevent browser container clipping */
    .stImage img {
        padding-bottom: 60px !important;
        object-fit: contain !important;
        max-height: 75vh !important;
    }
    hr {
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Garbage Classification: Architecture Comparison")

try:
    df = pd.read_csv(RESULTS_FILE)
    
    st.header("🏆 Experiment Leaderboard")
    sorted_df = df.sort_values(by='Val_Acc', ascending=False).reset_index(drop=True)
    st.dataframe(sorted_df) 
    
    st.header("Model Evaluation Visualizations")
    
    model_choice = st.selectbox("Select a completed run:", sorted_df.apply(
        lambda x: f"{x['Model']}_{x['Optimizer']}_{x['LR']}_{x['Epochs']}_{x['Batch_Size']}", axis=1
    ))
    
    if model_choice:
        # Displaying one entire image after another vertically
        st.write("---")
        st.image(os.path.join(OUTPUT_DIR, f"{model_choice}_history.png"), caption="Training History", use_container_width=True)
        
        st.write("---")
        st.image(os.path.join(OUTPUT_DIR, f"{model_choice}_confusion_matrix.png"), caption="Confusion Matrix", use_container_width=True)
        
        st.write("---")
        st.image(os.path.join(OUTPUT_DIR, f"{model_choice}_metrics.png"), caption="Final Metrics", use_container_width=True)
        
        st.write("---")
        st.image(os.path.join(OUTPUT_DIR, f"{model_choice}_classification_report.png"), caption="Classification Report", use_container_width=True)

except FileNotFoundError:
    st.error("Could not find results.csv. Please ensure the file is in your local directory.")