import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- Streamlit Page Config ---
st.set_page_config(page_title="SVM Breast Cancer Prediction", layout="wide")
st.title("🔬 Breast Cancer Prediction using SVM")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean-data.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True, errors='ignore')
    return df

data = load_data()
st.markdown("### Data Preview")
st.dataframe(data.head())

# --- Preprocessing ---
X = data.iloc[:, 1:31].values
y = LabelEncoder().fit_transform(data.iloc[:, 0].values)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=2, stratify=y)

# --- SVM Model ---
clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
st.success(f"✅ Classifier accuracy on test set: **{accuracy:.2f}**")

# --- Cross-Validation with Feature Selection ---
clf2 = make_pipeline(SelectKBest(f_regression, k=3), SVC(probability=True))
cv_score = cross_val_score(clf2, Xs, y, cv=3).mean()
st.info(f"🔁 3-Fold Cross-Validation Accuracy: **{cv_score:.2f}**")

# --- Confusion Matrix ---
st.markdown("### Confusion Matrix")
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(1.6, 1.2))  
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    square=True,
    cbar=False,
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant'],
    ax=ax_cm,
    annot_kws={"size": 6}  # Box number font
)
ax_cm.set_xlabel("Predicted", fontsize=5)  # Smaller axis labels
ax_cm.set_ylabel("Actual", fontsize=5)
ax_cm.set_title("Confusion Matrix", fontsize=6)
ax_cm.tick_params(axis='x', labelsize=5)
ax_cm.tick_params(axis='y', labelsize=5)
st.pyplot(fig_cm)

# --- Classification Report ---
st.markdown("### Classification Report")
report_text = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
st.text(report_text)

# --- ROC Curve ---
st.markdown("### ROC Curve")
probas = clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3))  # Reduced ROC size
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontsize=10)
ax_roc.set_ylabel('True Positive Rate', fontsize=10)
ax_roc.set_title('ROC Curve', fontsize=12)
ax_roc.legend(loc='lower right', fontsize=9)
ax_roc.tick_params(axis='both', labelsize=9)
ax_roc.grid(True)
st.pyplot(fig_roc)
