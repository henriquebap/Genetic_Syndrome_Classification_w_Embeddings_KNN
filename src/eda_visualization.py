import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE


def visualize_syndrome_distribution(df, output_dir):
    # COunt images per sydrome and plot bar chart - Conta imagens por sindrome e plota um grafico de barras
    syndrome_counts = df["syndrome_id"].value_counts()
    plt.figure(figsize=(10, 6))
    syndrome_counts.plot(kind="bar", color="skyblue")
    plt.title("Number of Images per Syndrome")
    plt.xlabel("Syndrome ID")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=90)
    plt.savefig(f"{output_dir}/syndrome_distribution.png")
    plt.close()


def tsne_visualization(df, output_dir):
    # Convert embeddings to array and perform t-SNE - Converte os 'embeddings' para array e realiza o t-SNE
    embeddings = np.array(df["embedding"].tolist())
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1)
    )
    tsne_results = tsne.fit_transform(embeddings)
    df["tsne_x"], df["tsne_y"] = tsne_results[:, 0], tsne_results[:, 1]

    # Plot t-SNE results - Plota os resultados do t-SNE
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="tsne_x", y="tsne_y", hue="syndrome_id", data=df, palette="tab10")
    plt.title("t-SNE Visualization of Embeddings")
    plt.savefig(f"{output_dir}/tsne_visualization.png")
    plt.close()


def tsne_density_heatmap(df, output_dir):
    # Perform t-SNE and create a density heatmap - Realiza o t-SNE e cria um 'heatmap'
    plt.figure(figsize=(12, 8))
    sns.kdeplot(
        x="tsne_x",
        y="tsne_y",
        hue="syndrome_id",
        data=df,
        fill=True,
        palette="tab10",
        alpha=0.5,
        levels=10,
    )
    plt.title("t-SNE Cluster Density Heatmap")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Syndrome ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(f"{output_dir}/tsne_cluster_density_heatmap.png")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("../data/outputs/preprocessed_data.csv")
    visualize_syndrome_distribution(df, "../data/outputs/plots")
    tsne_visualization(df, "../data/outputs/plots")

# Interpretacao dos cluester, funcionamento do t-SNE e padroes de visualizacao:

# O t-SNE reduz os dados de 320 dimensões para 2 dimensões, ajudando a visualizar como os embeddings se relacionam.
# Cada ponto no gráfico representa uma imagem, colorido de acordo com a síndrome correspondente.
# Clusters mostram imagens com características parecidas, o que ajuda na classificação.
# Clusters distintos indicam que os dados têm boas diferenças para separar as síndromes.
# Clusters sobrepostos podem significar que algumas síndromes têm características muito parecidas, dificultando a classificação.
# A diferença de tamanho dos clusters pode mostrar que algumas síndromes têm menos dados disponíveis.


# t-SNE reduces data from 320 dimensions to 2 dimensions, making it easier to see relationships between embeddings.
# Each point on the graph represents an image, colored based on its corresponding syndrome.
# Clusters show images with similar features, which helps with classification.
# Distinct clusters indicate that the data has clear differences to separate syndromes.
# Overlapping clusters may mean some syndromes share similar features, making classification harder.
# The size of the clusters can reveal that some syndromes have fewer data samples available.
