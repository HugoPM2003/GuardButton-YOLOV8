import sys
import os
import cv2
import json
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

# Diretórios
DATA_DIR = "data"
POSES_DIR = "poses"
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
MODEL_PATH = "yolo11n-pose.pt"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(POSES_DIR, exist_ok=True)
if not os.path.exists(LABELS_CSV):
    pd.DataFrame(columns=["video", "label"]).to_csv(LABELS_CSV, index=False)

model = YOLO(MODEL_PATH)

class GuardButtonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GuardButton")
        self.setGeometry(100, 100, 960, 720)

        self.video_path = ""
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

        font_label = QFont("Arial", 10)

        self.label = QLabel("Nenhum vídeo carregado")
        self.label.setMinimumSize(880, 600)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 1px solid gray;")
        self.label.setScaledContents(True)

        self.video_name = QLabel("Vídeo: --")
        self.video_name.setFont(font_label)
        self.video_name.setAlignment(Qt.AlignCenter)

        self.btn_load = QPushButton("Selecionar Vídeo")
        self.btn_load.setFont(QFont("Arial", 12))
        self.btn_load.clicked.connect(self.load_video)
        self.btn_load.setStyleSheet("background-color: #3498db; color: white; padding: 8px;")

        self.btn_roubo = QPushButton("[1] Roubo")
        self.btn_roubo.setFont(QFont("Arial", 12))
        self.btn_roubo.clicked.connect(lambda: self.classify_video("roubo"))
        self.btn_roubo.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px;")

        self.btn_normal = QPushButton("[2] Normal")
        self.btn_normal.setFont(QFont("Arial", 12))
        self.btn_normal.clicked.connect(lambda: self.classify_video("normal"))
        self.btn_normal.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px;")

        self.btn_incerto = QPushButton("[3] Incerto")
        self.btn_incerto.setFont(QFont("Arial", 12))
        self.btn_incerto.clicked.connect(lambda: self.classify_video("incerto"))
        self.btn_incerto.setStyleSheet("background-color: #f1c40f; color: white; padding: 8px;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.video_name)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_roubo)
        btn_layout.addWidget(self.btn_normal)
        btn_layout.addWidget(self.btn_incerto)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Selecionar vídeo", "", "Vídeos (*.mp4)")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.video_name.setText(f"Vídeo: {os.path.basename(self.video_path)}")
            self.timer.start(30)

    def play_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(q_img))
            else:
                self.cap.release()
                self.timer.stop()

    def classify_video(self, label):
        if not self.video_path:
            QMessageBox.warning(self, "Erro", "Selecione um vídeo primeiro!")
            return

        filename = os.path.basename(self.video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pose_output_path = os.path.join(POSES_DIR, f"{filename.replace('.mp4', '')}_{timestamp}.json")

        # Extrai keypoints com YOLOv11
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        pose_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, save=False, conf=0.3, verbose=False)
            keypoints = results[0].keypoints.xy.cpu().tolist() if results and results[0].keypoints else []
            pose_data.append({"frame": frame_count, "keypoints": keypoints})
            frame_count += 1

        cap.release()

        with open(pose_output_path, "w") as f:
            json.dump(pose_data, f, indent=2)

        # Salva rótulo
        df = pd.read_csv(LABELS_CSV)
        df = pd.concat([df, pd.DataFrame([{"video": filename, "label": label}])], ignore_index=True)
        df.to_csv(LABELS_CSV, index=False)

        QMessageBox.information(self, "Sucesso", f"Classificação '{label}' salva e keypoints extraídos!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GuardButtonApp()
    window.show()
    sys.exit(app.exec_())
