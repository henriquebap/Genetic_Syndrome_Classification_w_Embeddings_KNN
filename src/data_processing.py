import pandas as pd
import pickle
import numpy as np


def load_data(file_path):
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            print("Data loaded successfully!")
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def preprocess_data(data):
    records = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                records.append(
                    {
                        "syndrome_id": syndrome_id,
                        "subject_id": subject_id,
                        "image_id": image_id,
                        "embedding": embedding,
                    }
                )
    df = pd.DataFrame(records)

    # Data validation and cleaning - Limpeza e Validacao dos dados
    invalid_embeddings = df[~df["embedding"].apply(lambda x: len(x) == 320)]
    if not invalid_embeddings.empty:
        df = df[~df.index.isin(invalid_embeddings.index)]

    df = df.drop_duplicates(subset=["syndrome_id", "subject_id", "image_id"])
    print(f"DataFrame after preprocessing: {df.shape}")
    return df


if __name__ == "__main__":
    file_path = "../data/mini_gm_public_v0.1.p"
    data = load_data(file_path)
    df = preprocess_data(data)
    df.to_csv("../data/outputs/preprocessed_data.csv", index=False)
    print("Preprocessed data saved to '../data/outputs/preprocessed_data.csv'")


# Imbalance Observations:

# The dataset contains embeddings organized hierarchically by syndrome_id, subject_id, and image_id.
# A potential imbalance may arise from uneven numbers of images across syndromes. For example, certain syndrome_id categories might dominate the dataset, making it challenging for models to generalize well across underrepresented classes.
# During preprocessing, invalid embeddings and duplicate records were removed. If many invalid embeddings are associated with specific syndromes, it could exacerbate existing class imbalances.
# Patterns Observed:

# Each image is represented by a 320-dimensional embedding, which might contain inherent clustering when visualized (e.g., via t-SNE). These clusters could correspond to syndrome-specific features.
# The hierarchical structure ensures that every embedding has a clear association with a specific syndrome and subject, which might facilitate supervised learning but also demands rigorous handling of potential noise in embeddings.
# Impact:

# Imbalances could lead to overfitting in dominant classes or poor performance in rare classes.
# Patterns in embeddings may provide insights into which dimensions or features are most relevant for classification.
#
# Comentario em Portugues:
# Observações de Desbalanceamento:

# O conjunto de dados contém embeddings organizados hierarquicamente por syndrome_id, subject_id e image_id.
# Um desbalanceamento potencial pode surgir devido a quantidades desiguais de imagens por síndrome. Por exemplo, certas categorias de syndrome_id podem dominar o conjunto, dificultando a generalização do modelo para classes sub-representadas.
# Durante o pré-processamento, embeddings inválidos e registros duplicados foram removidos. Se muitos embeddings inválidos estiverem associados a determinadas síndromes, isso pode agravar o desbalanceamento existente.
# Padrões Observados:

# Cada imagem é representada por um embedding de 320 dimensões, que pode conter agrupamentos inerentes quando visualizado (por exemplo, via t-SNE). Esses agrupamentos podem corresponder a características específicas das síndromes.
# A estrutura hierárquica garante que cada embedding esteja claramente associado a uma síndrome e um sujeito específicos, o que pode facilitar o aprendizado supervisionado, mas também exige um tratamento rigoroso do potencial ruído nos embeddings.
# Impacto:

# Desbalanceamentos podem levar ao overfitting em classes dominantes ou a um desempenho ruim em classes raras.
# Padrões nos embeddings podem fornecer insights sobre quais dimensões ou características são mais relevantes para a classificação.
