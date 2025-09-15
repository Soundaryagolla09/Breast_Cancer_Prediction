import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Config ---
st.set_page_config(
    page_title="Breast Cancer Data EDA & Visualization",
    page_icon="📊",
    layout="wide"
)

# --- Title Section ---
st.markdown("""
    <div style="text-align:center; padding-top: 20px;">
        <h1 style="color:#5B9AA0;">📊 Breast Cancer Data Exploration & Visualization</h1>
        <h4 style="color:#7D7D7D;">Understanding cleaned dataset with descriptive stats and plots</h4>
    </div>
""", unsafe_allow_html=True)

# --- Load Data ---
data_path = "data/clean-data.csv"
if not os.path.exists(data_path):
    st.error(f"❌ Cleaned data file not found at `{data_path}`. Please upload and clean data first.")
    st.stop()

data = pd.read_csv(data_path)
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# --- Plotting helper ---
def st_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

# --- Overview Section ---
st.markdown("---")
st.markdown("### 🗂️ Dataset Preview")
st.dataframe(data.head(3))

with st.expander("📊 Statistical Overview"):
    st.write("### Description")
    st.write(data.describe())
    st.write("### 📈 Numerical Feature Skewness")
    st.write(data.select_dtypes(include='number').skew())
    st.write("### 🧬 Diagnosis Labels")
    st.write(data['diagnosis'].unique())
    st.write("### 🔢 Diagnosis Counts")
    st.write(data['diagnosis'].value_counts())

# --- Feature Grouping ---
data_mean = data.iloc[:, 1:11]
data_se = data.iloc[:, 11:22]
data_worst = data.iloc[:, 22:]

# --- Visualization Options ---
st.markdown("---")
viz_option = st.selectbox("Choose visualization to display", [
    "Diagnosis Countplot",
    "Histograms",
    "Density Plots",
    "Boxplots",
    "Correlation Heatmap",
    "Pairplot"
])

if viz_option == "Diagnosis Countplot":
    st.markdown("### 🧬 Diagnosis Frequency")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x='diagnosis', palette="Set3", ax=ax)
    ax.set_title("Diagnosis Countplot")
    st_plot(fig)

elif viz_option == "Histograms":
    for name, subset in zip(["Mean", "SE", "Worst"], [data_mean, data_se, data_worst]):
        st.markdown(f"### 📊 Histograms of {name} Features")
        subset.hist(bins=10, grid=False, figsize=(15, 10))
        fig = plt.gcf()
        st_plot(fig)

elif viz_option == "Density Plots":
    for name, subset in zip(["Mean", "SE", "Worst"], [data_mean, data_se, data_worst]):
        st.markdown(f"### 📈 Density Plots of {name} Features")
        subset.plot(kind='density', subplots=True, layout=(4, 3), sharex=False,
                    sharey=False, fontsize=9, figsize=(10, 6))
        fig = plt.gcf()
        st_plot(fig)

elif viz_option == "Boxplots":
    for name, subset in zip(["Mean", "SE", "Worst"], [data_mean, data_se, data_worst]):
        st.markdown(f"### 📦 Boxplots of {name} Features")
        subset.plot(kind='box', subplots=True, layout=(4, 4), sharex=False,
                    sharey=False, fontsize=9, figsize=(10, 6))
        fig = plt.gcf()
        st_plot(fig)

elif viz_option == "Correlation Heatmap":
    st.markdown("### 🔗 Feature Correlation Heatmap (Mean Features)")
    corr = data_mean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(260, 10, as_cmap=True),
                annot=True, fmt=".2g", linewidths=1, ax=ax, square=True)
    ax.set_title("Correlation Matrix")
    st_plot(fig)

elif viz_option == "Pairplot":
    st.markdown("### 🔍 Pairwise Relationships (Selected Features)")
    selected_cols = list(data.columns[1:7]) + ['diagnosis']
    pair_data = data[selected_cols]
    g = sns.pairplot(pair_data, hue='diagnosis', palette="Set1", diag_kind='hist', height=1.8)
    plt.tight_layout()
    st.pyplot(g.fig)
    plt.close(g.fig)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:#A9A9A9; font-size: 15px;'>
        <em>Visual exploration helps you understand your data better for improved modeling!</em>
    </div>
""", unsafe_allow_html=True)
