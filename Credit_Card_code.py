import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, davies_bouldin_score, silhouette_score,
                             calinski_harabasz_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import shap

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Load and Preprocess the Data
def load_and_preprocess():
    data = pd.read_csv("CC GENERAL.csv")
    data.dropna(inplace=True)
    
    # Exclude Customer ID and the target column "PURCHASES" from features
    feature_columns = [col for col in data.columns if col not in ['CUST_ID', 'PURCHASES']]
    X = data[feature_columns]
    
    # Handle missing values using KNN Imputer
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_imputed = imputer.fit_transform(X)
    
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return data, X_scaled, feature_columns

# 2. Dimensionality Reduction using PCA
def perform_pca(X_scaled):
    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
    optimal_components = np.argmax(explained_variance >= 0.95) + 1  # retain 95% variance
    pca = PCA(n_components=optimal_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA optimal components: {optimal_components}")
    return pca, X_pca

# 3. Supervised Learning (Behavior Prediction)
def behavior_prediction(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define base models for stacking
    base_models = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('XGBoost', XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(random_state=42))
    ]
    
    # Meta-model stacking classifier
    stacking_classifier = StackingClassifier(
        estimators=base_models, 
        final_estimator=RandomForestClassifier(random_state=42)
    )
    
    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'RandomForest__n_estimators': [100],
        'XGBoost__learning_rate': [0.01],
        'GradientBoosting__n_estimators': [100]
    }
    
    random_search = RandomizedSearchCV(
        stacking_classifier, param_dist, cv=3, scoring='roc_auc', n_jobs=-1, n_iter=5,
        random_state=42, verbose=2
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"\nBest Behavior Model: {best_model}")
    
    return best_model, X_train, X_test, y_train, y_test

# 4. Unsupervised Learning (Risk/Segmentation)
def risk_segmentation(X_pca):
    clusters = {
        'KMeans': KMeans(n_clusters=3, random_state=42),
        'GaussianMixture': GaussianMixture(n_components=3, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
    }
    
    best_cluster_model = None
    best_db_score = float('inf')
    
    print("\nClustering Evaluation Metrics:")
    for name, model in clusters.items():
        try:
            if hasattr(model, "fit_predict"):
                cluster_labels = model.fit_predict(X_pca)
            else:
                model.fit(X_pca)
                cluster_labels = model.predict(X_pca)
                
            if len(set(cluster_labels)) > 1:
                db_score = davies_bouldin_score(X_pca, cluster_labels)
                sil_score = silhouette_score(X_pca, cluster_labels)
                calinski_score = calinski_harabasz_score(X_pca, cluster_labels)
                print(f"{name} - Davies-Bouldin: {db_score:.4f}, Silhouette: {sil_score:.4f}, Calinski-Harabasz: {calinski_score:.4f}")
                if db_score < best_db_score:
                    best_db_score = db_score
                    best_cluster_model = model
            else:
                print(f"{name} produced only one cluster, skipping metrics.")
                
        except Exception as e:
            print(f"Error in {name}: {e}")
            continue
    
    print(f"\nBest Risk (Clustering) Model: {best_cluster_model}")
    return best_cluster_model

# 5. Anomaly Detection
def anomaly_detection(X_scaled):
    # Using Isolation Forest
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_labels = isolation_forest.fit_predict(X_scaled)
    num_anomalies_iso = np.sum(iso_labels == -1)
    print(f"\nIsolationForest detected anomalies: {num_anomalies_iso}")
    
    # Using Autoencoder
    input_dim = X_scaled.shape[1]
    encoding_dim = 8
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    ae_labels = (mse > threshold).astype(int)
    num_anomalies_ae = np.sum(ae_labels)
    print(f"Autoencoder detected anomalies: {num_anomalies_ae}")
    
    return iso_labels, ae_labels

# 6. Visualizations
def create_visualizations(best_model, X_test, y_test, feature_columns, best_cluster_model, X_pca, behavior_preds_all, risk_preds, data):
    # Classification Report
    y_behavior_pred = best_model.predict(X_test)
    print("\nClassification Report (Behavior Prediction):")
    print(classification_report(y_test, y_behavior_pred))
    
    # SHAP Analysis
    if "XGBoost" in best_model.named_estimators_:
        xgb_model = best_model.named_estimators_["XGBoost"]
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=feature_columns)
    else:
        print("Warning: XGBoost not found in the selected model. Skipping SHAP analysis.")
    
    # ROC Curve and Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        axes[0].plot([0, 1], [0, 1], linestyle='--')
        axes[0].set_title("Behavior - ROC Curve")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "predict_proba not available", ha="center", va="center")
    
    cm = confusion_matrix(y_test, y_behavior_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title("Behavior - Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    # Cluster Visualization
    if best_cluster_model._class.name_ == "DBSCAN":
        db_labels = best_cluster_model.fit_predict(X_pca)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap='viridis', s=50, alpha=0.7)
        plt.title("DBSCAN Clustering Results (PCA Projection)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster Label")
        plt.show()
    else:
        print("The selected best clustering model is not DBSCAN.")
    
    # Combined Visualization
    spending_intensity = data["PURCHASES"].values
    colors = np.where(behavior_preds_all == 1, 'red', 'blue')
    plt.figure(figsize=(12, 6))
    customer_indices = np.arange(len(data))
    plt.scatter(customer_indices, risk_preds, c=colors, s=spending_intensity/10, alpha=0.7, edgecolors='k')
    plt.xlabel("Customer Index")
    plt.ylabel("Risk Cluster (-1 indicates anomalies)")
    plt.title("Combined Visualization: Behavior Prediction & Risk Clustering")
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High Spender'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Low Spender')
    ]
    plt.legend(handles=legend_elements, title="Spending Behavior")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

# Main Execution
def main():
    # 1. Load and preprocess data
    data, X_scaled, feature_columns = load_and_preprocess()
    
    # 2. Dimensionality reduction
    pca, X_pca = perform_pca(X_scaled)
    
    # 3. Behavior prediction
    y = (data["PURCHASES"] > np.median(data["PURCHASES"])).astype(int)
    best_model, X_train, X_test, y_train, y_test = behavior_prediction(X_scaled, y)
    
    # 4. Risk segmentation
    best_cluster_model = risk_segmentation(X_pca)
    
    # Get risk predictions
    if hasattr(best_cluster_model, "fit_predict"):
        risk_preds = best_cluster_model.fit_predict(X_pca)
    else:
        best_cluster_model.fit(X_pca)
        risk_preds = best_cluster_model.predict(X_pca)
    
    # 5. Anomaly detection
    iso_labels, ae_labels = anomaly_detection(X_scaled)
    
    # 6. Get behavior predictions for all data
    behavior_preds_all = best_model.predict(X_scaled)
    
    # 7. Create visualizations
    create_visualizations(
        best_model, X_test, y_test, feature_columns, 
        best_cluster_model, X_pca, behavior_preds_all, 
        risk_preds, data
    )
    
    # 8. Sample predictions
    random_indices = np.random.choice(len(X_test), 5, replace=False)
    random_samples = X_test[random_indices]
    behavior_preds_samples = best_model.predict(random_samples)
    random_samples_pca = pca.transform(random_samples)
    
    if hasattr(best_cluster_model, "fit_predict"):
        risk_preds_samples = best_cluster_model.fit_predict(random_samples_pca)
    else:
        best_cluster_model.fit(random_samples_pca)
        risk_preds_samples = best_cluster_model.predict(random_samples_pca)
    
    print("\nSample Predictions:")
    for i, idx in enumerate(random_indices):
        beh = "High Spender" if behavior_preds_samples[i] == 1 else "Low Spender"
        print(f"Customer {idx}: Behavior Prediction = {beh}, Risk Cluster = {risk_preds_samples[i]}")

if _name_ == "_main_":
    main()