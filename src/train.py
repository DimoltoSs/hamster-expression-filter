"""Entrena el clasificador de expresiones (red neuronal en Keras) a partir de los CSV de landmarks.

Uso:
    python src/train.py
    python src/train.py --data-dir expresiones --epochs 200 --split temporal

Guarda models/modelo_hamsters.keras, models/clases.npy y en results/ las curvas, la matriz de confusión y metrics.json.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from data import load_raw, split_indices
from features import features_from_values


def build_model(X_train, n_classes, dropout=0.3):
    from tensorflow.keras import layers, models

    # Estandariza cada entrada (media 0, desviación 1); queda guardada dentro del modelo.
    norm = layers.Normalization()
    norm.adapt(X_train)
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        norm,
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(history.history["accuracy"], label="Entrenamiento")
    axes[0].plot(history.history["val_accuracy"], label="Validación")
    axes[0].set(title="Precisión del modelo", xlabel="Época", ylabel="Accuracy", ylim=(0, 1))
    axes[1].plot(history.history["loss"], label="Entrenamiento")
    axes[1].plot(history.history["val_loss"], label="Validación")
    axes[1].set(title="Pérdida del modelo", xlabel="Época", ylabel="Loss")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    nombres = [c.replace("hamster", "") for c in classes]
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=nombres).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set(title="Matriz de confusión (validación)", xlabel="Predicción", ylabel="Real")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="expresiones", help="carpeta con los CSV de landmarks")
    parser.add_argument("--out", default="models", help="carpeta donde se guardan modelo y clases")
    parser.add_argument("--results-dir", default="results", help="carpeta para curvas y métricas")
    parser.add_argument("--epochs", type=int, default=200, help="máximo de épocas (EarlyStopping corta antes)")
    parser.add_argument("--patience", type=int, default=15, help="épocas sin mejorar antes de parar")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras import callbacks

    tf.keras.utils.set_random_seed(args.seed)

    raw, labels, frames = load_raw(args.data_dir)
    X = features_from_values(raw)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype("int32")
    classes = [str(c) for c in encoder.classes_]

    tr, va = split_indices(labels, frames, mode=args.split, seed=args.seed)
    print("--- CONTEO DE CLASES ---")
    for c in classes:
        print(f"  {c:22s} {np.sum(labels == c)}")
    print(f"\nEntrenamiento: {len(tr)} | Validación: {len(va)} ({args.split}) | Clases: {len(classes)}")

    model = build_model(X[tr], len(classes))
    history = model.fit(
        X[tr], y[tr],
        validation_data=(X[va], y[va]),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
    )

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    model_path = os.path.join(args.out, "modelo_hamsters.keras")
    model.save(model_path)
    np.save(os.path.join(args.out, "clases.npy"), encoder.classes_)

    pred = model.predict(X[va], verbose=0).argmax(axis=1)
    report = classification_report(y[va], pred, target_names=classes, digits=3, output_dict=True, zero_division=0)
    cm = confusion_matrix(y[va], pred, labels=range(len(classes)))
    save_curves(history, os.path.join(args.results_dir, "training_curves.png"))
    save_confusion(cm, classes, os.path.join(args.results_dir, "confusion_matrix.png"))

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    with open(os.path.join(args.results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "epochs_run": len(history.history["loss"]),
            "best_epoch": best_epoch,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "classes": classes,
            "val_accuracy": round(float(report["accuracy"]), 4),
            "per_class_f1": {c: round(report[c]["f1-score"], 3) for c in classes},
        }, f, indent=2, ensure_ascii=False)

    print("\n" + classification_report(y[va], pred, target_names=classes, digits=3, zero_division=0))
    print(f"Mejor época: {best_epoch} | Accuracy de validación: {report['accuracy']:.4f}")
    print(f"Modelo guardado en {model_path}")


if __name__ == "__main__":
    main()
