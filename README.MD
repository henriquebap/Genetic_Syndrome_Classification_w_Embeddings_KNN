# Genetic Syndrome Classification with Embeddings

🚀 **Projeto de Machine Learning** - Classificação de Síndromes Genéticas usando embeddings de imagens.

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como parte de um teste técnico para uma vaga de **Desenvolvedor de Machine Learning**. O objetivo principal é analisar embeddings extraídos de imagens e utilizá-los para **classificar síndromes genéticas**.

Os embeddings são **vetores de 320 dimensões** gerados por um modelo pré-treinado, e o desafio é criar uma pipeline robusta para:

✔ **Pré-processamento dos dados**
✔ **Análise exploratória** (EDA)
✔ **Visualização de dados com t-SNE**
✔ **Classificação usando KNN**
✔ **Avaliação de métricas manuais**
✔ **Interpretação e análise dos resultados**

---

## 📂 Estrutura do Projeto

```bash
├── data/                     # Diretório de dados
│   ├── mini_gm_public_v0.1.p # Arquivo original de embeddings (pickle)
│   ├── preprocessed_data.csv # Dados processados para análise
│   ├── outputs/              # Resultados do modelo
│   │   ├── models/           # (Reservado para modelos treinados, se necessário)
│   │   ├── plots/            # Visualizações geradas pelo projeto
│   │   │   ├── confusion_matrix_cosine.png
│   │   │   ├── confusion_matrix_euclidean.png
│   │   │   ├── knn_accuracy_cosine.png
│   │   │   ├── knn_accuracy_euclidean.png
│   │   │   ├── roc_curve_comparison.png
│   │   │   ├── syndrome_distribution.png
│   │   │   ├── tsne_cluster_density_heatmap.png
│   │   │   ├── tsne_visualization.png
│   │   ├── tables/           # Tabelas de métricas e relatórios de performance
│   │   │   ├── classification_report_cosine.txt
│   │   │   ├── classification_report_euclidean.txt
│   │   │   ├── knn_results_summary.csv
│   │   │   ├── performance_metrics_summary.csv
├── src/                      # Código-fonte do projeto
│   ├── data_processing.py     # Script para carregamento e pré-processamento
│   ├── eda_visualization.py   # Análise exploratória e visualização de dados
│   ├── KNN_Classification.py  # Implementação do KNN para classificação
│   ├── metrics_evaluation.py  # Cálculo e avaliação de métricas de performance
│   ├── main.py                # Pipeline principal
├── README.md                  # Documentação do projeto
├── requirements.txt           # Bibliotecas necessárias
└── report.pdf                 # Relatório técnico final
```

---

## 📊 **Dataset**

O dataset é fornecido no formato **pickle** (`mini_gm_public_v0.1.p`) e segue a estrutura hierárquica:

```python
{
  'syndrome_id': {
    'subject_id': {
      'image_id': [320-dimensional embedding]
    }
  }
}
```

O dataset original **não está incluído neste repositório** por questões de privacidade e direitos de uso.

🎯 **Objetivo** : Usar esses embeddings para classificar a síndrome genética associada a cada imagem.

📌 **Possível erro ao carregar o arquivo** :

Se encontrar o erro `"numpy.core._multiarray_umath"`, atualize o NumPy com:

```bash
pip install --upgrade numpy
```

---

## **Pipeline do Projeto**

### 🔹 1. Pré-processamento dos Dados (`data_processing.py`)

- Carregamento e conversão dos embeddings do formato **pickle** para um **DataFrame**
- Normalização dos dados
- Tratamento de valores ausentes

### 🔹 2. Análise Exploratória e Visualização (`eda_visualization.py`)

- Estatísticas gerais: **quantidade de síndromes, distribuição de imagens por síndrome**
- Identificação de **desbalanceamento de classes**
- Geração dos gráficos:
  - **`syndrome_distribution.png`** → Distribuição das síndromes
  - **`tsne_visualization.png`** → Visualização 2D dos embeddings com t-SNE
  - **`tsne_cluster_density_heatmap.png`** → Heatmap de densidade dos clusters

### 🔹 3. Classificação (`KNN_Classification.py`)

- Implementação do **K-Nearest Neighbors (KNN)**
- Uso das **métricas de distância Cosine e Euclidean**
- **10-Fold Cross-Validation** para encontrar o melhor `k` (1 a 15)
- Geração dos gráficos:
  - **`knn_accuracy_cosine.png`** e **`knn_accuracy_euclidean.png`** → Precisão do modelo
  - **`confusion_matrix_cosine.png`** e **`confusion_matrix_euclidean.png`** → Matrizes de confusão

### 🔹 4. Avaliação de Métricas (`metrics_evaluation.py`)

- Cálculo das métricas de performance:
  ✔ **Área sob a curva ROC (AUC)**
  ✔ **F1-Score**
  ✔ **Top-k Accuracy**
- Geração das **Curvas ROC** :
  - **`roc_curve_comparison.png`** → Comparação entre as métricas de distância
- Relatórios e tabelas geradas:
  - **classification_report_cosine.txt** e **classification_report_euclidean.txt**
  - **knn_results_summary.csv**
  - **performance_metrics_summary.csv**

## 📈 **Resultados**

✔ A **visualização com t-SNE** revelou padrões de agrupamento úteis para classificação.

✔ O modelo **KNN** teve melhor desempenho com **k=5** usando **distância Cosine** .

✔ A métrica **AUC e F1-Score** indicaram boa separabilidade entre as classes.

✔ A curva ROC mostrou diferenças entre as métricas de distância.

📌 **Insights principais** :

- A distância **Cosine** obteve **melhor desempenho** do que a Euclidean.
- Algumas síndromes possuem dados desbalanceados, impactando a classificação.
- Melhorias futuras podem incluir **redes neurais e data augmentation** .

---

## 💻 **Como Executar o Projeto**

### 1️⃣ Clonar o repositório:

```bash
git clone https://github.com/seu-usuario/genetic-syndrome-classification.git
cd genetic-syndrome-classification
```

### 2️⃣ Instalar as dependências:

```bash
pip install -r requirements.txt
```

### 3️⃣ Executar a pipeline completa:

```bash
python src/main.py
```

---

## 📄 **Relatório**

Um relatório técnico completo sobre os resultados e interpretações do modelo pode ser encontrado em **`report.pdf`** .

---

## 📬 **Contato**

📌 Desenvolvido por **Henrique Baptista**

📧 Email: [henriquebaptista2003@gmail.com](mailto:henriquebaptista2003@gmail.com)

🔗 GitHub: [henriquebap](https://github.com/henriquebap)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/henrique-baptista777/)

---
