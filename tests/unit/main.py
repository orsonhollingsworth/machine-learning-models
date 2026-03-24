# main.py

import os
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from utils import load_data, prepare_data, load_config

def main():
    parser = argparse.ArgumentParser(description='Run the machine learning model')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    X, y = load_data(config['data_path'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=42)

    step1 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    step2 = Pipeline([
        ('pca', PCA(n_components=config['pca_components']))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('step1', step1, config['numerical_features']),
            ('step2', step2, config['numerical_features'])
        ]
    )

    model = RandomForestClassifier(n_estimators=config['n_estimators'], random_state=42)
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()