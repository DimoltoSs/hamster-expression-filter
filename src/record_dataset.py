"""Graba muestras de una expresión con la cámara y las guarda como CSV de landmarks.

Uso:
    python src/record_dataset.py --label hamsterfeliz --frames 300

Se abre una ventana con la cámara. Pon la expresión, presiona ESPACIO para empezar a grabar y 'q' para cancelar.
Genera ``expresiones/dataset_expresiones_<label>.csv`` con una fila por cuadro (etiqueta + 1434 valores x, y, z).
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp

from features import N_LANDMARKS, landmarks_to_values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="nombre de la clase, por ejemplo hamsterfeliz")
    parser.add_argument("--frames", type=int, default=300, help="cuadros a grabar")
    parser.add_argument("--out-dir", default="expresiones")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"dataset_expresiones_{args.label}.csv")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {args.camera}")

    header = ["clase"] + [f"{eje}{i}" for i in range(1, N_LANDMARKS + 1) for eje in "xyz"]
    recording, saved = False, 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while saved < args.frames:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            resultado = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hay_rostro = bool(resultado.multi_face_landmarks)

            if recording and hay_rostro:
                writer.writerow([args.label] + landmarks_to_values(resultado.multi_face_landmarks[0]))
                saved += 1

            estado = f"Grabando {saved}/{args.frames}" if recording else "ESPACIO: grabar | q: cancelar"
            if not hay_rostro:
                estado += " (sin rostro)"
            cv2.putText(frame, f"{args.label} - {estado}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Grabar expresion", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord(" "):
                recording = True
            elif tecla == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} cuadros guardados en {out_path}")


if __name__ == "__main__":
    main()
