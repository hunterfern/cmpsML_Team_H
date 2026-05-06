import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config import FIGURE_DIR, OUTPUT_DIR

def evaluate_sklearn_model(model, X_train, X_test, y_train, y_test, model_name, cv_folds=5):
    print(f"\n{'='*50}")
    print(f"  {model_name} - Model Assessment")
    print(f"{'='*50}\n")


    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous'])}")
    print(f"Confusion Matrix:\n{cm}")

    #Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
    print(f"\n{cv_folds}-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Save confusion matrix
    _save_confusion_matrix(cm, model_name)

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "CV Mean Accuracy": round(cv_scores.mean(), 4),
        "CV Std": round(cv_scores.std(), 4)
    }

def evaluate_ann_model(model, X_test, y_test):
    print(f"\n{'='*50}")
    print(f"  ANN - Model Assessment")
    print(f"{'='*50}")

    # ANN ouputs probabilities for each class, argmax picks highest one
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous'])}")
    print(f"Confusion Matrix:\n{cm}")

    _save_confusion_matrix(cm, "ANN")

    return {
        "Model"    : "ANN",
        "Accuracy" : round(acc, 4),
        "Precision": round(precision, 4),
        "Recall"   : round(recall, 4),
        "F1-Score" : round(f1, 4),
        "CV Mean Accuracy" : "N/A",
        "CV Std"   : "N/A",
    }

def _save_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Edible', 'Poisonous'],
        yticklabels=['Edible', 'Poisonous'],
        ax=ax
    )

    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    save_path = FIGURE_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")

def generate_comparison_report(results):
    (OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print("  All Models - Comparison Summary")
    print(f"{'='*50}")
    print(df.to_string(index=False))

    csv_path = OUTPUT_DIR / "metrics" / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nComparison report saved to {csv_path}")

    # accuracy bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(df["Model"], df["Accuracy"], color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison_accuracy.png")
    plt.close(fig)

    # Precision, Recall, F1 grouped chart
    metrics = ["Precision", "Recall", "F1-Score"]
    x = np.arange(len(df["Model"]))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        vals = pd.to_numeric(df[metric], errors='coerce')
        rects = ax.bar(x + i * width, vals, width, label=metric)
        ax.bar_label(rects, fmt="%.3f", padding=2, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Model"])
    ax.set_title("Model Precision, Recall, F1 Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison_metrics.png")
    plt.close(fig)

    print("Comparison charts saved to DOC/figures/")
    return df
