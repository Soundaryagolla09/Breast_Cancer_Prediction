import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8,4)

st.title("Breast Cancer Classification with SVC and PCA")

# Upload CSV or load default
uploaded_file = st.file_uploader("Upload CSV dataset (clean-data.csv format)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset file 'data/clean-data.csv'")
    data = pd.read_csv('data/clean-data.csv')

if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Show raw data if checkbox checked
if st.checkbox("Show raw data"):
    st.dataframe(data)

# Preprocess data
X = data.iloc[:, 1:31].values
y = data.iloc[:, 0].values

le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

st.write(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Build pipeline
pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('svc', SVC(probability=True))
])

# Cross-validation before tuning
cv = KFold(n_splits=10, shuffle=True, random_state=7)
cv_scores = cross_val_score(pipe_svc, X_train, y_train, cv=cv, scoring='accuracy')
st.write(f"Initial CV accuracy (train): {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# Hyperparameter tuning section
st.subheader("Hyperparameter Tuning")

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}
]

if st.button("Run Grid Search"):
    with st.spinner("Running GridSearchCV... this may take a moment"):
        gs = GridSearchCV(pipe_svc, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        
        st.success("Grid search completed!")
        st.write(f"Best CV score: {gs.best_score_:.4f}")
        st.write("Best parameters:")
        st.json(gs.best_params_)
        
        # Save the best estimator in session state
        st.session_state['best_model'] = gs.best_estimator_

# Use best model if available, else default pipeline
model_to_use = st.session_state.get('best_model', pipe_svc)
model_to_use.fit(X_train, y_train)

# Final CV score on training set
final_cv_scores = cross_val_score(model_to_use, X_train, y_train, cv=cv, scoring='accuracy')
st.write(f"Final CV accuracy (train): {np.mean(final_cv_scores):.3f} ± {np.std(final_cv_scores):.3f}")

# Test set evaluation
y_pred = model_to_use.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
st.write(f"Test set accuracy: {test_acc:.4f}")

# Confusion Matrix plot
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Classification report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Save model button
if st.button("Save Trained Model"):
    joblib.dump(model_to_use, "breast_cancer_model.pkl")
    st.success("Model saved as breast_cancer_model.pkl")

