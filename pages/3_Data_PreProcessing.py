import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# --- Page Config ---
st.set_page_config(
    page_title="Breast Cancer Data Analysis",
    page_icon="📊",
    layout="wide"
)

# --- Title and Intro ---
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#5B9AA0;">📊 Breast Cancer Data Analysis & PCA</h1>
        <h4 style="color:#7D7D7D;">Exploratory Data Analysis and Principal Component Analysis</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/clean-data.csv')
    # Safely drop 'Unnamed: 0' if present
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
    return df

data = load_data()

st.markdown("### Dataset Preview")
st.dataframe(data.head())

# --- Label Encoding ---
le = LabelEncoder()
data['diagnosis_encoded'] = le.fit_transform(data['diagnosis'])

# --- Show Class Distribution ---
st.markdown("### Class Distribution")
st.bar_chart(data['diagnosis'].value_counts())

# --- PCA Computation ---
X = data.iloc[:, 1:31].values
y = data['diagnosis_encoded'].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(Xs)

# Explained variance
explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# --- PCA Scatter Plot ---
st.markdown("### PCA Scatter Plot (First 2 Components)")

fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=data['diagnosis'],
    palette={"M": "red", "B": "blue"},
    alpha=0.7,
    ax=ax
)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.legend(title="Diagnosis")
st.pyplot(fig)

# --- Cumulative Variance Plot ---
st.markdown("### Cumulative Explained Variance by PCA Components")

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(range(1, len(cum_var)+1), cum_var*100, marker='o', color="#5B9AA0")
ax2.set_title("Cumulative Explained Variance (%)")
ax2.set_xlabel("Number of Principal Components")
ax2.set_ylabel("Cumulative Variance Explained (%)")
ax2.grid(True)
st.pyplot(fig2)

# --- Scree Plot ---
st.markdown("### Scree Plot")

fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.plot(range(1, len(explained_var)+1), explained_var, marker='o', linestyle='-', color="#D46A6A")
ax3.set_title("Scree Plot")
ax3.set_xlabel("Principal Component")
ax3.set_ylabel("Variance Explained")
ax3.grid(True)
st.pyplot(fig3)

# --- Summary Text ---
st.markdown("---")
st.markdown(
    """
    <div style="padding: 15px;">
        <h4>Summary:</h4>
        <p>
            The dataset contains features derived from breast cancer tumors. The PCA reduces
            dimensionality while retaining variance. The first two principal components provide
            a clear separation between malignant (red) and benign (blue) tumors. The cumulative
            variance plot shows how many components explain most of the variance, which is helpful
            for feature reduction in downstream modeling.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
