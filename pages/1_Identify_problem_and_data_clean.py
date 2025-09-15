import streamlit as st
import pandas as pd
import io

# --- Page Config ---
st.set_page_config(
    page_title="Identify Problem & Clean Data",
    page_icon="🔍",
    layout="wide"
)

# --- Title Section ---
st.markdown(
    """
    <div style="text-align:center; padding-top: 20px;">
        <h1 style="color:#5B9AA0;">🔍 Problem Identification & Data Cleaning</h1>
        <h4 style="color:#7D7D7D;">Upload your dataset and prepare it for analysis</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Upload Section ---
st.markdown("---")
st.markdown("### 📁 Upload Your Breast Cancer Dataset (.csv format)")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and read successfully!")

        # --- Raw Preview ---
        st.markdown("### 🗂️ Raw Dataset Preview")
        st.dataframe(data.head(2))

        # --- Drop Redundant Columns ---
        if "id" in data.columns:
            data.drop("id", axis=1, inplace=True)
            st.info("Dropped column: `id`")

        # --- Dataset Shape ---
        st.markdown("### 📏 Dataset Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

        # --- Dataset Info ---
        st.markdown("### ℹ️ Dataset Info")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

        # --- Null Check ---
        st.markdown("### ❓ Null Values")
        nulls = data.isnull().sum()
        if nulls.any():
            st.dataframe(nulls[nulls > 0])
        else:
            st.success("✅ No missing values found.")

        # --- Diagnosis Labels ---
        if "diagnosis" in data.columns:
            st.markdown("### 🧬 Diagnosis Labels")
            st.write(data["diagnosis"].unique())

        # --- Save Cleaned Data ---
        st.markdown("### 💾 Save Cleaned Data")
        cleaned_file_path = "data/clean-data.csv"
        data.to_csv(cleaned_file_path, index=False)
        st.success(f"✅ Cleaned data saved at: `{cleaned_file_path}`")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")

else:
    st.info("Please upload a CSV file to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#A9A9A9; font-size: 15px;'>
        <em>Data cleaned today leads to predictions you can trust tomorrow.</em>
    </div>
    """,
    unsafe_allow_html=True
)
