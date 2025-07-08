# 🧠 Credit Card Customer Segmentation & Behavior Prediction

This project applies a combination of **Supervised Learning**, **Unsupervised Learning**, and **Anomaly Detection** techniques to classify customers based on their credit card usage, segment them into risk profiles, and detect unusual behavior.

---

## 📂 Dataset

- **Source**: `CC GENERAL.csv`
- **Target**: `PURCHASES` column (used to define high vs low spenders)

---

## 🚀 Features

- 🧼 **Data Preprocessing**  
  - Dropping `CUST_ID`
  - KNN Imputation of missing values
  - Standard Scaling

- 📉 **Dimensionality Reduction**  
  - PCA with 95% explained variance

- ✅ **Supervised Learning: Behavior Prediction**  
  - Models: `RandomForest`, `XGBoost`, `SVM`, `GradientBoosting`
  - Final Model: `StackingClassifier` with `RandomSearchCV`

- 📊 **Unsupervised Learning: Risk Segmentation**  
  - Algorithms: `KMeans`, `GaussianMixture`, `DBSCAN`
  - Evaluation: `Davies-Bouldin`, `Silhouette`, `Calinski-Harabasz`

- ⚠️ **Anomaly Detection**  
  - Techniques: `IsolationForest`, `Autoencoder (Keras)`
  - Threshold: Top 5% MSE from autoencoder

- 🔍 **Explainability with SHAP**
- 📈 **Visualizations**: ROC Curve, Confusion Matrix, Clusters, Combined View

##  SHAP Analysis
<img width="363" alt="Screenshot 2025-07-08 222856" src="https://github.com/user-attachments/assets/5bd14310-3697-4b92-8a14-77c1634bfd98" />

## ROC Curve and Confusion Matrix of Proposed Model 
<img width="346" alt="Screenshot 2025-07-08 222831" src="https://github.com/user-attachments/assets/8df32aac-99c1-4cca-b260-637ef736dd37" />

## PCA Projection 
<img width="455" alt="Screenshot 2025-07-08 222723" src="https://github.com/user-attachments/assets/ef4e1af7-ac07-49ef-bcef-7851d7f97ca0" />

## Behavior Prediction & Risk Clustering 
<img width="324" alt="Screenshot 2025-07-08 222651" src="https://github.com/user-attachments/assets/e38d59f9-3c8f-486c-8ff8-061a24df91bc" />
