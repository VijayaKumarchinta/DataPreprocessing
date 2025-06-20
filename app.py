import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path=None):
    try:
        if file_path:
            data = pd.read_csv(file_path)
        else:
            iris = load_iris()
            data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            data['target'] = iris.target
        logger.info("Dataset loaded successfully")
        print("\nDataset Info:")
        print(data.info())
        print("\nFirst 5 Rows:")
        print(data.head())
        print("\nSummary Statistics:")
        print(data.describe())
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def preprocess_data(data):
    try:
        np.random.seed(42)
        mask = np.random.choice([True, False], size=data.shape, p=[0.05, 0.95])
        data[mask] = np.nan
        print("\nMissing Values Before Imputation:")
        print(data.isnull().sum())
        numeric_cols = data.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        print("\nMissing Values After Imputation:")
        print(data.isnull().sum())

        Q1 = data[numeric_cols].quantile(0.25)
        Q3 = data[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        print(f"\nOutliers detected in {outlier_mask.sum()} rows")
        data = data[~outlier_mask].reset_index(drop=True)

        categorical_cols = data.select_dtypes(include=['object']).columns
        if categorical_cols.any():
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
            print("\nCategorical columns encoded with one-hot encoding")

        le = LabelEncoder()
        if 'target' in data.columns:
            data['target'] = le.fit_transform(data['target'])
            print("\nTarget Distribution:")
            print(data['target'].value_counts())

        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(data.drop('target', axis=1, errors='ignore'))
        scaled_df = pd.DataFrame(scaled_features, columns=data.columns.drop('target', errors='ignore'))
        if 'target' in data.columns:
            scaled_df['target'] = data['target']
        print("\nData after scaling:")
        print(scaled_df.head())

        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(scaled_df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(f'distribution_{col}.png')
            plt.close()
        logger.info("Distribution plots saved")
        return scaled_df, le
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None, None

def feature_engineering(data):
    try:
        data['sepal_length_width'] = data['sepal length (cm)'] * data['sepal width (cm)']
        data['petal_length_width'] = data['petal length (cm)'] * data['petal width (cm)']
        data['sepal_length_bin'] = pd.qcut(data['sepal length (cm)'], q=4, labels=False)
        data = pd.get_dummies(data, columns=['sepal_length_bin'], prefix='sepal_bin')

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(data.drop('target', axis=1, errors='ignore'))
        poly_feature_names = poly.get_feature_names_out(data.drop('target', axis=1, errors='ignore').columns)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
        if 'target' in data.columns:
            poly_df['target'] = data['target']

        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(poly_df.drop('target', axis=1, errors='ignore'))
        pca_df = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])
        if 'target' in data.columns:
            pca_df['target'] = poly_df['target']
        print("\nPCA Explained Variance Ratio:", pca.explained_variance_ratio_)

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 3), pca.explained_variance_ratio_.cumsum(), marker='o')
        plt.title('PCA Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig('pca_variance.png')
        plt.close()
        logger.info("PCA variance plot saved")

        X = poly_df.drop('target', axis=1, errors='ignore')
        y = poly_df['target'] if 'target' in poly_df.columns else None
        selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        print("\nSelected Features:", selected_features)

        final_df = pd.DataFrame(X_selected, columns=selected_features)
        if y is not None:
            final_df['target'] = y
        return final_df, selected_features
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return None, None

def train_and_evaluate(data, selected_features):
    try:
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("\nTest Set Accuracy:", accuracy)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("\nCross-Validation Accuracy: Mean =", cv_scores.mean(), "Std =", cv_scores.std())

        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.savefig('feature_importance.png')
        plt.close()
        logger.info("Feature importance plot saved")
        return model
    except Exception as e:
        logger.error(f"Error in model training/evaluation: {e}")
        return None

def save_dataset(data, filename='preprocessed_data.csv'):
    try:
        data.to_csv(filename, index=False)
        logger.info(f"Preprocessed dataset saved as '{filename}'")
        print(f"\nPreprocessed dataset saved as '{filename}'")
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")

def main(file_path=None):
    data = load_dataset(file_path)
    if data is None:
        return
    processed_data, label_encoder = preprocess_data(data)
    if processed_data is None:
        return
    final_data, selected_features = feature_engineering(processed_data)
    if final_data is None:
        return
    if 'target' in final_data.columns:
        model = train_and_evaluate(final_data, selected_features)
        if model is None:
            return
    save_dataset(final_data)

if __name__ == "__main__":
    main()