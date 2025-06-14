import os
import cv2
import numpy as np
from ultralytics import YOLO
import joblib
from config import VIDEO_FOLDER
# Configurações
FRAME_SKIP = 5
CONF_THRESHOLD = 0.3
ALERTA_CONF = 0.6  # probabilidade mínima para disparar alerta

# Carrega modelos
model_pose = YOLO("yolov8n-pose.pt")
clf = joblib.load("modelo_roubo_keypoints_rf.joblib")

# Função para extrair vetor flat dos keypoints
def keypoints_to_vector(keypoints):
    return np.array([coord for point in keypoints for coord in point]).reshape(1, -1)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = model_pose.predict(source=frame_rgb, conf=CONF_THRESHOLD, verbose=False)[0]

            if results_pose.keypoints is not None:
                for person in results_pose.keypoints.xy:
                    keypoints = person.tolist()
                    if len(keypoints) == 17:
                        vec = keypoints_to_vector(keypoints)
                        prob = clf.predict_proba(vec)[0][1]  # probabilidade de roubo

                        if prob >= ALERTA_CONF:
                            print(f"[ALERTA] Movimento suspeito detectado em {video_name} frame {frame_idx} (prob={prob:.2f})")
                            # Opcional: salvar frame com alerta, enviar notificação, etc.

        frame_idx += 1

    cap.release()

# Processa todos os vídeos
for file in os.listdir(VIDEO_FOLDER):
    if file.endswith(".avi"):
        video_path = os.path.join(VIDEO_FOLDER, file)
        print(f"[INFO] Processando vídeo: {file}")
        process_video(video_path)

print("[✓] Processamento finalizado.")
