"""Carga de los CSV de expresiones y división en entrenamiento y validación."""

import glob
import os

import numpy as np
import pandas as pd

from features import N_FEATURES


def load_raw(data_dir):
    """Devuelve (valores crudos (n, 1434), etiquetas (n,), número de cuadro dentro de su grabación (n,))."""
    paths = sorted(glob.glob(os.path.join(data_dir, "dataset_expresiones_*.csv")))
    if not paths:
        raise SystemExit(f"No encontré archivos dataset_expresiones_*.csv en '{data_dir}'")

    values, labels, frames = [], [], []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            has_header = "clase" in f.readline().lower()
        df = pd.read_csv(path, header=None, skiprows=1 if has_header else 0)
        v = df.iloc[:, -N_FEATURES:].apply(pd.to_numeric, errors="coerce").dropna().values.astype(np.float32)
        clase = os.path.basename(path).replace("dataset_expresiones_", "").replace(".csv", "")
        values.append(v)
        labels.extend([clase] * len(v))
        frames.append(np.arange(len(v)))
    return np.vstack(values), np.array(labels), np.concatenate(frames)


def split_indices(labels, frames, val_frac=0.2, mode="temporal", seed=42):
    """Índices (entrenamiento, validación).

    temporal: el último val_frac de cada grabación va a validación. Los cuadros seguidos son casi iguales, así que
    una división aleatoria pondría casi copias del entrenamiento en la validación e inflaría la precisión.
    """
    labels = np.asarray(labels)
    if mode == "random":
        from sklearn.model_selection import train_test_split
        return train_test_split(np.arange(len(labels)), test_size=val_frac, random_state=seed, stratify=labels)

    val = np.zeros(len(labels), dtype=bool)
    for clase in np.unique(labels):
        sel = labels == clase
        corte = int(np.ceil(frames[sel].max() * (1 - val_frac)))
        val[sel] = frames[sel] >= corte
    return np.where(~val)[0], np.where(val)[0]
