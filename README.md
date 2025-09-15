🎗️ Predicting Breast Cancer Diagnosis Using Machine Learning

By: Golla Soundarya, Aagam Shreeha, Vasanthavada Venkata Sushma Manojna
Department of CSE - Data Science
Institute of Aeronautical Engineering, Dundigal, Hyderabad, Telangana - 500043

📖 Abstract

Breast cancer is one of the most common malignancies among women and a leading cause of cancer-related deaths. Early and accurate diagnosis is critical for effective treatment and survival.

This project presents a machine learning-based predictive model to classify breast tumors as benign or malignant using the Wisconsin Breast Cancer Diagnostic Dataset. The model is trained on features extracted from Fine Needle Aspiration (FNA) tests and evaluated using algorithms like Support Vector Machine (SVM), Logistic Regression, and Random Forest.

The final model is deployed as a Streamlit web application to provide a user-friendly prediction system that assists clinicians and patients in timely diagnosis.

🔹 Features

✅ Predicts whether a breast tumor is benign (0) or malignant (1)

✅ Uses multiple ML algorithms (SVM, Logistic Regression, Random Forest)

✅ Achieves ~98% accuracy with SVM

✅ Real-time prediction through a Streamlit web app

✅ User-friendly interface for medical support

🛠️ Tech Stack

Python

Scikit-learn – ML modeling & evaluation

Pandas, NumPy – Data preprocessing

Matplotlib, Seaborn – Visualization

Streamlit – Web app deployment

Pickle – Model serialization

📊 Methodology

Dataset – Wisconsin Breast Cancer Diagnostic Dataset (569 records, 30 features).

Preprocessing – Data cleaning, feature scaling (StandardScaler), and label encoding.

Modeling –

Logistic Regression

Random Forest

Support Vector Machine (SVM) with GridSearchCV tuning

Evaluation Metrics – Accuracy, Confusion Matrix, Precision, Recall, F1-score.

Deployment – Final model serialized (.pkl) and integrated into Streamlit for real-time predictions.

🏗️ System Architecture

Frontend (Streamlit) → Collects user input (FNA test results).

Backend (ML Model) → Processes input using trained classifier.

Data Layer → Preprocessed dataset + serialized model files.

📈 Results

SVM Model Accuracy: ~98%

Precision/Recall (Malignant): >97%

Very few false negatives → critical for medical safety.

🚀 Future Work

🔹 Mobile app deployment for wider accessibility

🔹 Integration with deep learning for image-based cancer diagnosis

🔹 Patient-specific prediction using biomarkers and historical data

🔹 Integration with Electronic Health Records (EHRs)

📚 References

UCI Machine Learning Repository – Breast Cancer Dataset

American Cancer Society – Breast Cancer Facts & Figures

Cortes, C., Vapnik, V. “Support-Vector Networks,” Machine Learning, 1995.

Breiman, L. “Random Forests,” Machine Learning, 2001.

Scikit-learn Documentation

Streamlit Documentation
