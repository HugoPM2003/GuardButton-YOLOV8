import cv2
import os
from config import VIDEO_FOLDER, OUTPUT_FRAMES_FOLDER

os.makedirs(OUTPUT_FRAMES_FOLDER, exist_ok=True)


def process_videos():
    for video_file in os.listdir(VIDEO_FOLDER):
        if video_file.endswith(".avi"):
            video_path = os.path.join(VIDEO_FOLDER, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Pré-processamento (ex: redimensionamento)
                frame = cv2.resize(frame, (640, 640))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Salva o frame como imagem
                frame_filename = f"{video_file}_frame_{frame_count:05d}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_FRAMES_FOLDER, frame_filename), frame)
                frame_count += 1

            cap.release()
            print(f"Processado: {video_file} — {frame_count} frames extraídos")

process_videos()
