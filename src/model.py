from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def train_model(X, y):
    # Split data — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, 
          target_names=["False Positive", "Confirmed", "Candidate"],
          labels=[0, 1, -1]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, -1]))

def get_feature_importance(model, config):
    features = config["feature_columns"]
    importances = model.feature_importances_

    pairs = sorted(zip(features, importances), 
                   key=lambda x: x[1], reverse=True)

    print("\nFeature Importance:")
    for feature, importance in pairs:
        print(f"  {feature}: {importance:.4f}")