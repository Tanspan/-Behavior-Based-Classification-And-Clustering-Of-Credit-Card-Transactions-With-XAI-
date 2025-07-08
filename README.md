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
<img width="346" alt="Screenshot 2025-07-08 222831" src="https://github.com/user-attachments/assets/0bbb3919-4ee9-476a-a5b8-6ca262108a17" />

<img width="455" alt="Screen<img width="324" alt="Screenshot 2025-07-08 222651" src="https://github.com/user-attachments/assets/7c17268c-cca9-4544-8d82-7da55daa12ce" />

<img width="363" alt="Screenshot 2025-07-08 222856" src="https://github.com/user-attachments/assets/4e9a4178-b27e-4137-9718-70491b13aba0" />
