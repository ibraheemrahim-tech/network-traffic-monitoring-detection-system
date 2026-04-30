"""
Detection Module
Implements Random Forest (supervised) and Isolation Forest (unsupervised) detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest


def prepare_data(features_df, labels, test_size=0.3, binary=True):
    """
    Prepare data for detection by splitting and scaling.
    """
    X = features_df.values
    y = labels.copy()

    if binary:
        y = y.apply(lambda x: "BENIGN" if str(x).upper() == "BENIGN" else "MALICIOUS")

    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def run_random_forest(X_train, X_test, y_train, y_test):
    """
    Train and run Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=20
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    importances = model.feature_importances_

    return predictions, probabilities, model, importances


def run_isolation_forest(X_train, X_test, contamination=0.1):
    """
    Train and run Isolation Forest anomaly detector.
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    raw_predictions = model.predict(X_test)
    anomaly_scores = model.decision_function(X_test)

    # Map: 1 -> Normal, -1 -> Suspicious
    predictions = np.where(raw_predictions == 1, "Normal", "Suspicious")

    return predictions, anomaly_scores
