# Predicting Breast Cancer Diagnosis Using Machine Learning Algorithms

## 📌 Overview
This project implements a machine learning–based predictive model to classify breast tumors as **benign** or **malignant** using the **Wisconsin Breast Cancer Diagnostic Dataset**.  
The system leverages **Support Vector Machine (SVM)**, **Logistic Regression**, and **Random Forest** classifiers, with the best-performing model deployed as a **Streamlit web application** for real-time predictions.

## 🧠 Motivation
Breast cancer is one of the most common malignancies among women and a leading cause of cancer-related deaths. Early and accurate diagnosis significantly improves treatment outcomes. This project aims to assist clinicians and patients by providing a fast, reliable, and interpretable diagnostic tool based on **Fine Needle Aspiration (FNA)** test results.

---

## 🚀 Features
- Classification of tumors into:
  - **1 = Malignant (Cancerous)**
  - **0 = Benign (Not Cancerous)**
- Multiple ML algorithms tested: **SVM, Logistic Regression, Random Forest**
- Hyperparameter tuning with **GridSearchCV**
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Deployed as a **Streamlit** web application
- Near-instantaneous predictions from user input

---

## 📂 Dataset
- **Source:** [UCI Machine Learning Repository - Wisconsin Breast Cancer (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Records:** 569
- **Features:** 30 input features from FNA images
- **Target:** Diagnosis (`M` for malignant, `B` for benign)

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `streamlit`
- **Model Deployment:** Streamlit Web App
- **Serialization:** Pickle (`.pkl` files)

---

## 📊 Model Performance
| Model               | Accuracy | Precision (Malignant) | Recall (Malignant) |
|---------------------|----------|-----------------------|--------------------|
| SVM (Best Model)    | ~98%     | >97%                  | >97%               |
| Logistic Regression | ~96%     | >95%                  | >95%               |
| Random Forest       | ~97%     | >96%                  | >96%               |

✅ **SVM achieved the highest accuracy and lowest false negatives**, which is critical in medical diagnosis.

---

## 📌 System Architecture
1. **Frontend Layer:**  
   - Streamlit app for user interaction
2. **Backend Layer:**  
   - Trained ML model + preprocessing pipeline
3. **Data Layer:**  
   - Preprocessed input features and serialized model files

---

## 📥 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
