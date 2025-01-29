import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from data_processing import load_data, preprocess_data
from eda_visualization import (
    visualize_syndrome_distribution,
    tsne_visualization,
    tsne_density_heatmap,
)
from knn_classification import (
    evaluate_knn,
    plot_knn_results,
    calculate_roc_auc,
    save_knn_results_to_csv,
)
from metrics_evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics_summary,
    save_classification_report,
)
from sklearn.metrics import classification_report


def main():
    base_output_dir = "data/outputs"
    os.makedirs(f"{base_output_dir}/plots", exist_ok=True)
    os.makedirs(f"{base_output_dir}/tables", exist_ok=True)

    # Data Processing
    file_path = "data/mini_gm_public_v0.1.p"
    data = load_data(file_path)
    df = preprocess_data(data)
    df.to_csv(f"{base_output_dir}/preprocessed_data.csv", index=False)

    # Visualization
    visualize_syndrome_distribution(df, f"{base_output_dir}/plots")
    tsne_visualization(df, f"{base_output_dir}/plots")
    tsne_density_heatmap(df, f"{base_output_dir}/plots")

    # KNN Classification
    X = np.array(df["embedding"].tolist())
    y = df["syndrome_id"]
    euclidean_results = evaluate_knn(X, y, metric="euclidean")
    cosine_results = evaluate_knn(X, y, metric="cosine")
    plot_knn_results(euclidean_results, "euclidean", f"{base_output_dir}/plots")
    plot_knn_results(cosine_results, "cosine", f"{base_output_dir}/plots")
    save_knn_results_to_csv(
        euclidean_results, cosine_results, f"{base_output_dir}/tables"
    )

    # ROC Curves
    fpr_euclidean, tpr_euclidean, mean_auc_euclidean = calculate_roc_auc(
        X, y, metric="euclidean"
    )
    fpr_cosine, tpr_cosine, mean_auc_cosine = calculate_roc_auc(X, y, metric="cosine")

    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr_euclidean, tpr_euclidean, label=f"Euclidean AUC = {mean_auc_euclidean:.2f}"
    )
    plt.plot(fpr_cosine, tpr_cosine, label=f"Cosine AUC = {mean_auc_cosine:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title("ROC Curve Comparison: Euclidean vs. Cosine")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"{base_output_dir}/plots/roc_curve_comparison.png")
    plt.close()

    # Final Metrics and Confusion Matrices
    knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_euclidean.fit(X, y)
    y_pred_euclidean = knn_euclidean.predict(X)
    y_proba_euclidean = knn_euclidean.predict_proba(X)
    metrics_euclidean = compute_metrics(y, y_pred_euclidean, y_proba_euclidean)
    plot_confusion_matrix(
        y, y_pred_euclidean, np.unique(y), "euclidean", f"{base_output_dir}/plots"
    )
    save_classification_report(
        y, y_pred_euclidean, "euclidean", f"{base_output_dir}/tables"
    )

    knn_cosine = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn_cosine.fit(X, y)
    y_pred_cosine = knn_cosine.predict(X)
    y_proba_cosine = knn_cosine.predict_proba(X)
    metrics_cosine = compute_metrics(y, y_pred_cosine, y_proba_cosine)
    plot_confusion_matrix(
        y, y_pred_cosine, np.unique(y), "cosine", f"{base_output_dir}/plots"
    )
    save_classification_report(y, y_pred_cosine, "cosine", f"{base_output_dir}/tables")

    # Save Metrics Summary
    save_metrics_summary(metrics_euclidean, metrics_cosine, f"{base_output_dir}/tables")

    # Print Classification Reports
    print("Classification Report - Euclidean Distance")
    print(classification_report(y, y_pred_euclidean))
    print("\nClassification Report - Cosine Distance")
    print(classification_report(y, y_pred_cosine))


if __name__ == "__main__":
    main()
