"""Extracción y normalización de landmarks faciales, compartida por entrenamiento y demo.

MediaPipe Face Mesh (refine_landmarks=True) da 478 puntos (x, y, z) relativos a la imagen. Para que el modelo vea la
expresión y no la posición de la cabeza, cada rostro se centra, se escala por la distancia entre los ojos y se gira
para dejar los ojos horizontales.
"""

import numpy as np

N_LANDMARKS = 478
N_FEATURES = N_LANDMARKS * 3  # 1434

OJO_IZQ = 33   # esquina externa del ojo izquierdo
OJO_DER = 263  # esquina externa del ojo derecho


def normalize_points(points: np.ndarray) -> np.ndarray:
    """(478, 3) o (n, 478, 3) -> (n, 1434) normalizado."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 2:
        pts = pts[None]
    pts = pts - pts.mean(axis=1, keepdims=True)

    ojos = pts[:, OJO_DER, :2] - pts[:, OJO_IZQ, :2]
    dist = np.linalg.norm(ojos, axis=1)
    pts = pts / np.maximum(dist, 1e-6)[:, None, None]

    ang = np.arctan2(ojos[:, 1], ojos[:, 0])
    cos, sin = np.cos(-ang)[:, None], np.sin(-ang)[:, None]
    x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
    pts[:, :, 0] = x * cos - y * sin
    pts[:, :, 1] = x * sin + y * cos
    return pts.reshape(len(pts), -1)


def features_from_values(values) -> np.ndarray:
    """Valores crudos (x1, y1, z1, x2, ...) de uno o varios rostros -> (n, 1434) normalizado."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[1] < N_FEATURES:
        raise ValueError(f"Se esperaban {N_FEATURES} valores por rostro y llegaron {arr.shape[1]}")
    return normalize_points(arr[:, -N_FEATURES:].reshape(-1, N_LANDMARKS, 3))


def landmarks_to_values(face_landmarks) -> list:
    """Landmarks de MediaPipe -> lista cruda (x1, y1, z1, x2, ...), como se guarda en los CSV."""
    values = []
    for lm in face_landmarks.landmark:
        values.extend((lm.x, lm.y, lm.z))
    return values


def landmarks_to_features(face_landmarks) -> np.ndarray:
    """Landmarks de MediaPipe -> entrada del modelo (1, 1434)."""
    return features_from_values(landmarks_to_values(face_landmarks))
