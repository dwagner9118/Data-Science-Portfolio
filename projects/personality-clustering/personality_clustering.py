import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data(file_path):
    """Load personality dataset"""
    df = pd.read_csv(file_path)
    return df


def select_personality_columns(df):
    """Select Big Five personality columns"""
    prefixes = ("EXT", "EST", "AGR", "CSN", "OPN")

    personality_columns = [
        col for col in df.columns if col.startswith(prefixes)
    ]

    return df[personality_columns]


def preprocess_data(df):
    """Handle missing values and scale data"""

    imputer = SimpleImputer(strategy="mean")
    imputed_data = imputer.fit_transform(df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    return scaled_data


def run_pca(data, components=10):
    """Run PCA dimensionality reduction"""

    pca = PCA(n_components=components, random_state=42)
    pca_data = pca.fit_transform(data)

    return pca_data, pca


def evaluate_clusters(data):
    """Evaluate cluster sizes using elbow method"""

    inertias = []
    silhouette_scores = []

    k_range = range(2, 11)

    for k in k_range:

        model = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = model.fit_predict(data)

        inertias.append(model.inertia_)

        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

    results = pd.DataFrame({
        "k": list(k_range),
        "inertia": inertias,
        "silhouette_score": silhouette_scores
    })

    return results


def fit_kmeans(data, k=4):
    """Fit final clustering model"""

    model = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = model.fit_predict(data)

    return model, labels


def plot_pca_variance(pca):
    """Plot PCA explained variance"""

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    plt.figure()

    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")

    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")

    plt.show()


def plot_elbow(results):

    plt.figure()

    plt.plot(results["k"], results["inertia"], marker="o")

    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")

    plt.show()


def plot_clusters(pca_data, labels):

    plt.figure()

    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Personality Clusters")

    plt.show()


def main():

    # load dataset
    df = load_data("personality_data.csv")

    # select personality columns
    personality_data = select_personality_columns(df)

    # preprocess data
    processed_data = preprocess_data(personality_data)

    # run PCA
    pca_data, pca_model = run_pca(processed_data)

    # evaluate cluster sizes
    results = evaluate_clusters(pca_data)

    print(results)

    # choose cluster size
    model, labels = fit_kmeans(pca_data, k=4)

    # plots
    plot_pca_variance(pca_model)
    plot_elbow(results)
    plot_clusters(pca_data, labels)

    # save output
    df["cluster"] = labels

    df.to_csv("personality_clusters_output.csv", index=False)

    print("Clustering completed successfully")


if __name__ == "__main__":
    main()
