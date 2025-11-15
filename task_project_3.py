"""
Bank Marketing Decision Tree classifier

This script will:
- download the UCI Bank Marketing dataset from the linked GitHub repo
- preprocess numeric and categorical features
- train a DecisionTreeClassifier with a scikit-learn Pipeline
- report accuracy, classification report and show a confusion matrix

Usage (PowerShell):
python task_project_3.py

Requirements: see `requirements.txt` in same folder
"""
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


RAW_URL = (
    "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/"
    "main/Task%203/bank-additional/bank-additional-full.csv"
)


def load_data(url=RAW_URL):
    # dataset uses semicolon delimiter and fields are enclosed in double quotes
    df = pd.read_csv(url, sep=';', quotechar='"')
    return df


def preprocess_and_split(df, test_size=0.25, random_state=42):
    # Drop rows with missing target (none exist), and optional noisy fields
    X = df.drop(columns=['y'])
    y = df['y'].map({'yes': 1, 'no': 0})

    # Select numeric and categorical columns (based on dataset description)
    numeric_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                    'euribor3m', 'nr.employed']

    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # pipeline for numeric features
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


def build_pipeline(preprocessor, max_depth=None, random_state=42):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', tree)])
    return pipeline


def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:\n", report)

    # plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return pipeline


def main():
    print("Loading dataset from GitHub...")
    df = load_data()
    print("Dataset loaded. Shape:", df.shape)

    # Basic EDA
    print(df.head())
    print(df['y'].value_counts())

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)

    print("Building Decision Tree pipeline and training...")
    pipeline = build_pipeline(preprocessor)

    trained_pipeline = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

    # Save model
    out = 'decision_tree_bank_marketing.joblib'
    joblib.dump(trained_pipeline, out)
    print(f"Trained model saved to {out}")


if __name__ == '__main__':
    main()
