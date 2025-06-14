import cv2
import os
import numpy as np
import joblib
from ultralytics import YOLO
from config import VIDEO_FOLDER

# Pasta onde salvar frames suspeitos
OUTPUT_FOLDER = 'teste_suspeitos'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Carregar modelo de pose e classificador
pose_model = YOLO('yolov8n-pose.pt')
clf = joblib.load("modelo_roubo_keypoints_rf.joblib")

def normalizar_keypoints(flat_keypoints):
    coords = np.array(flat_keypoints).reshape(-1, 2)
    valid_coords = coords[(coords != 0).all(axis=1)]
    if len(valid_coords) == 0:
        return flat_keypoints
    center = valid_coords.mean(axis=0)
    coords -= center
    scale = np.linalg.norm(valid_coords.max(axis=0) - valid_coords.min(axis=0))
    if scale > 0:
        coords /= scale
    return coords.flatten().tolist()

cap = cv2.VideoCapture(VIDEO_FOLDER)
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model.predict(source=frame, conf=0.3, verbose=False)[0]

    if results.keypoints is not None:
        for person in results.keypoints.xy:
            keypoints = person.tolist()
            flat = [coord for pair in keypoints for coord in pair]
            if len(flat) == 34:
                vetor_norm = normalizar_keypoints(flat)
                pred = clf.predict([vetor_norm])[0]

                if pred == 1:
                    print(f"[ALERTA] Comportamento suspeito no frame {frame_index}")

                    # Salvar frame suspeito
                    path_img = os.path.join(OUTPUT_FOLDER, f"suspeito_{frame_index}.jpg")
                    cv2.imwrite(path_img, frame)

    frame_index += 1

cap.release()
print("Processamento finalizado.")
