from ultralytics import YOLO
import cv2
import os
import json

model = YOLO("yolo11n-pose.pt")
data_folder = "data"

# Procura vídeos .mp4 na pasta
videos = [f for f in os.listdir(data_folder) if f.endswith(".mp4")]

for video_file in videos:
    video_path = os.path.join(data_folder, video_file)
    pose_output_path = os.path.join(data_folder, video_file.replace(".mp4", ".json"))

    # Se já existe JSON, ignora
    if os.path.exists(pose_output_path):
        print(f"Já processado: {video_file}")
        continue

    print(f"Processando {video_file}...")
    cap = cv2.VideoCapture(video_path)
    pose_data = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=0.3, verbose=False)
        keypoints = results[0].keypoints.xy.cpu().tolist() if results and results[0].keypoints else []
        pose_data.append({"frame": frame_count, "keypoints": keypoints})
        frame_count += 1

    cap.release()

    # Salva os keypoints em JSON
    with open(pose_output_path, "w") as f:
        json.dump(pose_data, f, indent=2)

    print(f"Poses salvas em {pose_output_path}")
