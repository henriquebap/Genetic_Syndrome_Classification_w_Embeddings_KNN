from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_metrics(y_test, y_pred, y_proba, k=5):
    # Compute evaluation metrics for the model - Calcula métricas de avaliação para o modelo
    metrics = {
        "Accuracy": np.mean(y_test == y_pred),
        "Top-k Accuracy": top_k_accuracy_score(y_test, y_proba, k=k),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        "AUC": roc_auc_score(
            label_binarize(y_test, classes=np.unique(y_test)),
            y_proba,
            multi_class="ovr",
        ),
    }
    return metrics


def save_metrics_summary(euclidean_metrics, cosine_metrics, output_dir):
    # Save performance metrics for Euclidean and Cosine metrics to CSV - Salva métricas de desempenho para métricas Euclidean e Cosine em CSV
    summary = pd.DataFrame(
        [
            {
                "Metric": key,
                "Euclidean": euclidean_metrics[key],
                "Cosine": cosine_metrics[key],
            }
            for key in euclidean_metrics.keys()
        ]
    )
    summary.to_csv(f"{output_dir}/performance_metrics_summary.csv", index=False)
    print(
        f"Performance metrics summary saved to '{output_dir}/performance_metrics_summary.csv'"
    )


def plot_confusion_matrix(y_test, y_pred, labels, metric, output_dir):
    # Generate and save a confusion matrix plot - Gera e salva o gráfico da matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp.plot(cmap="Blues", ax=ax)
    plt.title(f"Confusion Matrix ({metric.capitalize()} Metric)")
    plt.savefig(f"{output_dir}/confusion_matrix_{metric}.png")
    plt.close()
    print(
        f"Confusion matrix for {metric} saved to '{output_dir}/confusion_matrix_{metric}.png'"
    )


def plot_roc_curve(y_test, y_proba, metric, output_dir):
    # Generate and save the ROC curve plot - Gera e salva o gráfico da curva ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title(f"ROC Curve ({metric.capitalize()} Metric)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve_{metric}.png")
    plt.close()


def save_classification_report(y_test, y_pred, metric_name, output_dir):
    # Save classification report to a text file - Salva o relatório de classificação em um arquivo de texto
    """Salva o relatório de classificação em um arquivo .txt."""
    report = classification_report(y_test, y_pred, output_dict=False)
    file_path = f"{output_dir}/classification_report_{metric_name}.txt"
    with open(file_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to '{file_path}'")
