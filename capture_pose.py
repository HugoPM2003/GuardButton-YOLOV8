from ultralytics import YOLO
import cv2
import json
import os
from datetime import datetime

model = YOLO("yolo11n-pose.pt")
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# Captura da webcam
cap = cv2.VideoCapture(0)

# Tempo de captura (em segundos) e FPS
event_duration = 10
fps = 20
frame_count = 0
pose_data = []

# Nome dos arquivos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join(save_dir, f"video_event_{timestamp}.mp4")
pose_filename = os.path.join(save_dir, f"poses_event_{timestamp}.json")

# Configura gravador de vídeo
width = int(cap.get(3))
height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

print("Capturando evento...")

while frame_count < event_duration * fps:
    ret, frame = cap.read()
    if not ret:
        break

    # Salva vídeo
    out.write(frame)

    # Aplica YOLO para extrair poses
    results = model.predict(source=frame, save=False, conf=0.3, verbose=False)
    if results:
        keypoints = results[0].keypoints.xy.cpu().tolist() if results[0].keypoints else []
        pose_data.append({"frame": frame_count, "keypoints": keypoints})

    # Exibe
    cv2.imshow("Captura", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Salva poses
with open(pose_filename, "w") as f:
    json.dump(pose_data, f, indent=2)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Evento salvo em: {video_filename}")
print(f"Poses salvas em: {pose_filename}")
