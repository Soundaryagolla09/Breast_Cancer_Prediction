import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# --- Streamlit Page Config ---
st.set_page_config(page_title="SVM Breast Cancer with PCA & GridSearch", layout="wide")
st.title("🔬 Breast Cancer Prediction with PCA and SVM Grid Search")

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

# --- PCA for dimensionality reduction ---
pca = PCA(n_components=10)
X_pca = pca.fit_transform(Xs)

st.markdown("### PCA Explained Variance Ratio")
st.bar_chart(pca.explained_variance_ratio_)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=2, stratify=y
)

# --- Grid Search for best SVM parameters ---
param_grid = {
    'C': np.logspace(-3, 2, 6),
    'gamma': np.logspace(-3, 2, 6),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid = GridSearchCV(
    estimator=SVC(probability=True),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0,
    return_train_score=True,
    refit=True
)

with st.spinner("Running Grid Search..."):
    grid.fit(X_train, y_train)

st.success(f"Best Parameters: {grid.best_params_}")
st.info(f"Best Cross-Validation Score: {grid.best_score_:.3f}")

best_clf = grid.best_estimator_

# --- Evaluate on Test Data ---
y_pred = best_clf.predict(X_test)
accuracy = best_clf.score(X_test, y_test)
st.success(f"Test Accuracy: {accuracy:.2f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    ax=ax_cm,
    cbar=False
)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# --- Classification Report ---
st.markdown("### Classification Report")
report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
st.text(report)

# --- ROC Curve ---
probas = best_clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots(figsize=(5, 3))
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
ax_roc.set_xlim([0, 1])
ax_roc.set_ylim([0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc='lower right')
ax_roc.grid(True)
st.pyplot(fig_roc)

# --- Visualizing Decision Boundaries on first two PCA features ---
st.markdown("### Decision Boundaries on first two PCA components")

X_train_2d = X_train[:, :2]

C = 1.0  # regularization parameter
svm_linear = SVC(kernel='linear', C=C).fit(X_train_2d, y_train)
svm_rbf = SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train_2d, y_train)
svm_poly = SVC(kernel='poly', degree=3, C=C).fit(X_train_2d, y_train)

# Prepare meshgrid
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial (degree 3) kernel']
clfs = [svm_linear, svm_rbf, svm_poly]

for i, clf in enumerate(clfs):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
    axes[i].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', cmap=ListedColormap(['#FF0000', '#00FF00']))
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('PCA Component 1')
    axes[i].set_ylabel('PCA Component 2')
    axes[i].set_xlim(xx.min(), xx.max())
    axes[i].set_ylim(yy.min(), yy.max())
    axes[i].set_xticks([])
    axes[i].set_yticks([])

st.pyplot(fig)
