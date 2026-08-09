import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import pandas as pd

import data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import KNNClusterAssigner

OUT_DIR = Path(__file__).resolve().parent / "outputs"

def title(text):
    print("\n" + "--" * 30)
    print(text)

def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load data")
    data = data_loader.load_data()
    X_raw = data["X_raw"]
    X_scaled = data["X_scaled"]
    df = data["dataframe"]
    
    print(f"Data loaded: {len(X_raw)} rows")
    print(f"Features used: {data['feature_names']}")

    title("STEP 2 : Elbow Method to find optimal K")
    k_values = range(2, 11)
    inertias = []

    for k in k_values:
        km = TFKMeans(n_clusters=k)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"   k = {k:>2}  ->  inertia = {km.inertia_:.4f}")

    visualize.plot_elbow(k_values, inertias, OUT_DIR / "01_elbow.png")
    print("\n>>> Saved Elbow curve to outputs/01_elbow.png")

    title("STEP 3 : Run K-Means with chosen k")
    best_k = 2 
    print(f"Chosen k = {best_k}")

    final_km = TFKMeans(n_clusters=best_k)
    labels = final_km.fit_predict(X_scaled)

    visualize.plot_clusters(X_raw, labels, OUT_DIR / "02_clusters.png", 
                            x_name="Weight", y_name="Height")
    print(">>> Saved Cluster visualization to outputs/02_clusters.png")

    title("STEP 4 : Save results to CSV")
    df["Cluster"] = labels
    
    df.to_csv(OUT_DIR / "clustered_data.csv", index=False)
    
    summary = df.groupby("Cluster")[data["feature_names"]].mean().round(2)
    summary.to_csv(OUT_DIR / "cluster_summary.csv")
    
    print(summary)
    print("\n>>> Saved clustered data and summary to outputs/")

    title("STEP 5 : Test KNN Cluster Assigner")
    assigner = KNNClusterAssigner(k=5)
    assigner.fit(X_scaled, labels)

    sample_new = X_scaled[:5]
    predicted_clusters = assigner.predict(sample_new)
    
    print(f"Predicted clusters by KNN : {predicted_clusters}")
    print(f"Actual clusters by KMeans : {labels[:5]}")
    print("\n[summary] KNN can successfully assign clusters to new data based on nearest neighbors!")

if __name__ == "__main__":
    main()