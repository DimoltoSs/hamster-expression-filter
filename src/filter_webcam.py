"""Filtro en vivo: reconoce tu expresión con la cámara y muestra el hamster correspondiente.

Uso:
    python src/filter_webcam.py
    python src/filter_webcam.py --camera 1 --suavizado 10 --umbral 0.7

Presiona 'q' para salir.
"""

import argparse
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from features import landmarks_to_features

# Clase -> imagen en figuras/. Las clases que no están aquí (como "neutral") solo muestran texto.
FIGURAS = {
    "hamsteraura": "hamsteraura.png",
    "hamsterfeliz": "hamstermuyfeliz.jpg",
    "hamsterlibroabierto": "hamsterlibro.png",
    "hamsterlupa": "hamsterlupa.png",
    "hamstermrbean": "hamstermrbean.png",
    "hamsterpaleta": "hamsterpaleta.jpg",
    "hamsterenojado": "hamsterenojado.jpg",
    "hamsterasombrado": "hamsterAsombrado.jpg",
    "hamsternobrain": "hamsterNobrain.jpg",
}

TAMANO_MINIATURA = 150
MARGEN = 15


def overlay_image_alpha(img, img_overlay, x, y):
    """Pega img_overlay sobre img en (x, y), respetando la transparencia si la tiene."""
    if img_overlay.shape[2] == 4:
        alpha_mask = img_overlay[:, :, 3] / 255.0
        img_rgb = img_overlay[:, :, :3]
    else:
        alpha_mask = np.ones(img_overlay.shape[:2])
        img_rgb = img_overlay

    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    crop = img[y1:y2, x1:x2]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    for c in range(3):
        crop[:, :, c] = alpha * img_rgb[y1o:y2o, x1o:x2o, c] + (1.0 - alpha) * crop[:, :, c]
    return img


def load_figuras(figuras_dir):
    cargadas = {}
    for clase, archivo in FIGURAS.items():
        ruta = os.path.join(figuras_dir, archivo)
        img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED) if os.path.exists(ruta) else None
        if img is not None:
            img = cv2.resize(img, (TAMANO_MINIATURA, TAMANO_MINIATURA))
        cargadas[clase] = img
    faltan = [c for c, i in cargadas.items() if i is None]
    if faltan:
        print(f"Aviso: sin imagen para {', '.join(faltan)} (se mostrará solo el texto).")
    return cargadas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--figuras", default="figuras", help="carpeta con las imágenes de los hamsters")
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara")
    parser.add_argument("--suavizado", type=int, default=8, help="cuadros que se promedian (1 = sin suavizado)")
    parser.add_argument("--umbral", type=float, default=0.6, help="confianza mínima para mostrar un hamster (0 a 1)")
    args = parser.parse_args()

    from tensorflow.keras.models import load_model

    model_path = os.path.join(args.models_dir, "modelo_hamsters.keras")
    if not os.path.exists(model_path):
        raise SystemExit(f"No encontré '{model_path}'. Entrena el modelo con: python src/train.py")
    modelo = load_model(model_path)
    clases = np.load(os.path.join(args.models_dir, "clases.npy"), allow_pickle=True)
    figuras = load_figuras(args.figuras)

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {args.camera}")
    print("Filtro activo. Presiona 'q' en la ventana para salir.")

    # Promediar las últimas predicciones evita que el hamster cambie en cada cuadro.
    historial = deque(maxlen=max(1, args.suavizado))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        resultado = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ancho = frame.shape[1]

        clase, prob = None, 0.0
        if resultado.multi_face_landmarks:
            entrada = landmarks_to_features(resultado.multi_face_landmarks[0])
            historial.append(modelo(entrada, training=False).numpy()[0])
            promedio = np.mean(historial, axis=0)
            idx = int(np.argmax(promedio))
            clase, prob = str(clases[idx]), float(promedio[idx]) * 100
        else:
            historial.clear()

        if clase is not None and prob < args.umbral * 100:
            clase = f"? {clase}"  # poca confianza: solo texto

        if clase is not None:
            x0, y0 = ancho - TAMANO_MINIATURA - MARGEN, MARGEN
            if figuras.get(clase) is not None:
                frame = overlay_image_alpha(frame, figuras[clase], x0, y0)
            texto = f"{clase} ({prob:.1f}%)"
            (ancho_texto, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x_texto = max(10, ancho - ancho_texto - MARGEN)
            cv2.putText(frame, texto, (x_texto, y0 + TAMANO_MINIATURA + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Filtro IA de Hamsters", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
