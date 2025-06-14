import cv2
import os
import json
from ultralytics import YOLO
from datetime import datetime
from config import VIDEO_PATH, OUTPUT_FOLDER

# Caminhos
VIDEO_FOLDER = 'videos_entrada/furto_loja_01.avi'
OUTPUT_FOLDER = 'movimentos_confirmados/confirmados_pelo_segurança'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configurações
FRAME_SKIP = 5
MODEL_PATH = 'yolov8n-pose.pt'
model = YOLO(MODEL_PATH)

# Variáveis globais
click_position = None
selected_keypoints = None
selected_frame = None
selected_frame_index = 0

def mouse_callback(event, x, y, flags, param):
    global click_position
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)
        print(f"[+] Clique registrado em: {click_position}")

def get_closest_person(click_pos, results):
    """Retorna os keypoints da pessoa mais próxima ao clique"""
    min_dist = float('inf')
    closest = None

    for i, person in enumerate(results.keypoints.xy):
        keypoints = person.tolist()
        center_x = sum([kp[0] for kp in keypoints if kp[0] > 0]) / len(keypoints)
        center_y = sum([kp[1] for kp in keypoints if kp[1] > 0]) / len(keypoints)

        dist = ((click_pos[0] - center_x)**2 + (click_pos[1] - center_y)**2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = keypoints

    return closest

def process_video():
    global selected_frame, selected_keypoints, selected_frame_index
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_index = 0

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % FRAME_SKIP == 0:
            display = frame.copy()
            cv2.imshow("Video", display)
            key = cv2.waitKey(30)

            if click_position:
                # Roda pose detection
                results = model.predict(source=frame, conf=0.3, verbose=False)[0]

                if results.keypoints is not None:
                    selected = get_closest_person(click_position, results)
                    if selected:
                        selected_keypoints = selected
                        selected_frame = frame.copy()
                        selected_frame_index = frame_index
                        break
                    else:
                        print("[-] Nenhuma pessoa detectada próxima ao clique.")

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

def salvar_keypoints():
    if not selected_keypoints or not selected_frame:
        print("[-] Nenhuma pessoa foi selecionada para salvar.")
        return

    base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Salva imagem
    img_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{timestamp}.jpg")
    cv2.imwrite(img_path, selected_frame)

    # Salva JSON
    json_data = {
        "video": base_name,
        "frame_index": selected_frame_index,
        "timestamp": timestamp,
        "keypoints": selected_keypoints
    }
    json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"[✓] Keypoints salvos em: {json_path}")

if __name__ == "__main__":
    process_video()
    salvar_keypoints()