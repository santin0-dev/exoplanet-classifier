import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(model, config):
    features = config["feature_columns"]
    importances = model.feature_importances_
    
    pairs = sorted(zip(features, importances),
                   key=lambda x: x[1], reverse=True)
    features_sorted, importances_sorted = zip(*pairs)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(importances_sorted), y=list(features_sorted))
    plt.title("Feature Importance — What Makes a Planet Real?")
    plt.xlabel("Importance Score")
    plt.ylabel("NASA Measurement")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.show()
    print("Saved: plots/feature_importance.png")

def plot_confusion_matrix(model, X_test, y_test):
    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, -1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["False Positive", "Confirmed", "Candidate"],
                yticklabels=["False Positive", "Confirmed", "Candidate"])
    plt.title("Confusion Matrix — Where Does the Model Get Confused?")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: plots/confusion_matrix.png")

def plot_label_distribution(df, config):
    label_map = {1: "Confirmed", 0: "False Positive", -1: "Candidate"}
    labels = df[config["label_column"]].map(label_map)

    plt.figure(figsize=(8, 5))
    sns.countplot(x=labels, hue=labels, palette="viridis", legend=False)
    plt.title("Label Distribution — What Did Kepler Find?")
    plt.xlabel("Classification")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/label_distribution.png", dpi=150)
    plt.show()
    print("Saved: plots/label_distribution.png")