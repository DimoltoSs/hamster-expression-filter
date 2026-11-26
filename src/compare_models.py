"""Compara varios clasificadores de scikit-learn con landmarks crudos y normalizados.

Uso:
    python src/compare_models.py
    python src/compare_models.py --split random

Guarda la tabla en results/model_comparison.csv.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_raw, split_indices
from features import features_from_values

MODELOS = {
    "Regresión logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=100, gamma="scale")),
    "PCA(50) + SVM": lambda: make_pipeline(StandardScaler(), PCA(50, random_state=0), SVC(C=100, gamma="scale")),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    "KNN (k=5)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "MLP (128, 64)": lambda: make_pipeline(
        StandardScaler(), MLPClassifier((128, 64), max_iter=500, early_stopping=True, random_state=0)
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="expresiones")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    raw, labels, frames = load_raw(args.data_dir)
    variantes = {
        "crudos": raw,
        "normalizados": features_from_values(raw),
    }
    tr, va = split_indices(labels, frames, mode=args.split)
    print(f"Muestras: {len(labels)} | Entrenamiento: {len(tr)} | Validación: {len(va)} ({args.split})\n")

    filas = []
    for nombre_feat, X in variantes.items():
        for nombre_modelo, crear in MODELOS.items():
            t0 = time.time()
            modelo = crear().fit(X[tr], labels[tr])
            pred = modelo.predict(X[va])
            fila = {
                "landmarks": nombre_feat,
                "modelo": nombre_modelo,
                "accuracy": round(accuracy_score(labels[va], pred), 4),
                "f1_macro": round(f1_score(labels[va], pred, average="macro"), 4),
                "segundos": round(time.time() - t0, 1),
            }
            filas.append(fila)
            print(f"{nombre_feat:13s} {nombre_modelo:22s} acc={fila['accuracy']:.3f}  f1={fila['f1_macro']:.3f}")

    tabla = pd.DataFrame(filas).sort_values("accuracy", ascending=False)
    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "model_comparison.csv")
    tabla.to_csv(out, index=False)
    print(f"\nTabla guardada en {out}")


if __name__ == "__main__":
    main()
