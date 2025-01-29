import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize


def evaluate_knn(X, y, metric, k_range=(1, 15)):
    # Evaluate KNN accuracy for different k values and a given metric
    # Avalia a acurácia do KNN para diferentes valores de k e uma métrica específica

    results = []
    for k in range(k_range[0], k_range[1] + 1):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)  # Initialize KNN
        scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")  # 10-fold CV
        results.append(
            {"k": k, "mean_accuracy": scores.mean(), "std_accuracy": scores.std()}
        )
    return results  # Return list of accuracy result
    # Retorna uma lista com a média e o desvio padrão da acurácia para cada k


def plot_knn_results(results, metric, output_dir):
    # Extract data for plotting
    # Gera um gráfico da acurácia média do KNN para diferentes valores de k, com barras de erro.

    ks = [res["k"] for res in results]
    accuracies = [res["mean_accuracy"] for res in results]
    std_devs = [res["std_accuracy"] for res in results]

    # Plot mean accuracy with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        ks, accuracies, yerr=std_devs, fmt="o-", label=f"{metric.capitalize()} Accuracy"
    )
    plt.title(f"KNN Accuracy vs. K ({metric.capitalize()} Metric)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.xticks(ks)
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{output_dir}/knn_accuracy_{metric}.png"
    )  # Salva o gráfico em um arquivo PNG. # Save plot to file

    plt.close()


# Calcula as curvas ROC e o AUC médio usando validação cruzada estratificada.
def calculate_roc_auc(X, y, metric, n_splits=10):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fprs, tprs, aucs = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train KNN with a fixed k - Treinando KNN com um K fixo
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train, y_train)
        y_proba = knn.predict_proba(X_test)

        # Binarize labels for multiclass ROC
        y_test_bin = label_binarize(
            y_test, classes=np.unique(y)
        )  # Binarize labels - Binarizacao das Labels
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        roc_auc = auc(fpr, tpr)

        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

    return (
        np.mean(fprs, axis=0),
        np.mean(tprs, axis=0),
        np.mean(aucs),
    )  # Retorna as médias de FPR, TPR e AUC. # Return averages


def save_knn_results_to_csv(euclidean_results, cosine_results, output_dir):
    # Combine results into a DataFrame
    # Salva os resultados do KNN (acurácia para Euclidean e Cosine) em um arquivo CSV.

    results_summary = pd.DataFrame(
        {
            "k": [res["k"] for res in euclidean_results],
            "Euclidean Accuracy": [res["mean_accuracy"] for res in euclidean_results],
            "Cosine Accuracy": [res["mean_accuracy"] for res in cosine_results],
        }
    )
    results_summary.to_csv(
        f"{output_dir}/knn_results_summary.csv", index=False
    )  # Save to CSV - Salva em CSV
    print(
        f"KNN results saved to '{output_dir}/knn_results_summary.csv'"
    )  # Exibe uma mensagem de confirmação após salvar o arquivo. # Confirmation


# Euclidiana: Boa para dados normalizados, mas fraca em alta dimensionalidade.
# Cosseno: Foca na direção, resistente a ruídos e escalas.
# Sendo assim, o modelo com distância cosseno conseguiu resultados ligeiramente melhores,
# pois foi mais eficaz em lidar com a alta sobreposição e baixa separabilidade dos dados,
# características que dificultam a classificação com métricas como a distância euclidiana.
# Desequilíbrio: Classes dominantes impactam mais, especialmente com k alto.

# English
# Euclidean: Works well with normalized data but weak in high dimensions.
# Cosine: Focuses on direction, robust to noise and scaling.
# Thus, the cosine distance model achieved slightly better results as it was more effective in handling the high overlap and low separability of the data,
# which pose challenges for metrics like Euclidean distance.
# Imbalance: Dominant classes affect results, especially with high k.
