import cv2
import pandas as pd
import os

data_folder = "data"
label_file = os.path.join(data_folder, "labels.csv")

# Cria arquivo de labels se não existir
if not os.path.exists(label_file):
    df = pd.DataFrame(columns=["video", "label"])
    df.to_csv(label_file, index=False)

# Lista vídeos ainda não rotulados
videos = [f for f in os.listdir(data_folder) if f.endswith(".mp4")]
labeled_videos = pd.read_csv(label_file)["video"].tolist()
unlabeled_videos = [v for v in videos if v not in labeled_videos]

for video in unlabeled_videos:
    path = os.path.join(data_folder, video)
    cap = cv2.VideoCapture(path)

    print(f"Mostrando vídeo: {video}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Classifique o evento - [1] Roubo, [2] Não Roubo, [3] Incerto", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Entrada do usuário
    label = input("Classifique o evento: [1] Roubo  [2] Não Roubo  [3] Incerto: ")
    label_map = {"1": "roubo", "2": "nao_roubo", "3": "incerto"}
    label_str = label_map.get(label, "incerto")

    # Salva no CSV
    df = pd.read_csv(label_file)
    df = pd.concat([df, pd.DataFrame([{"video": video, "label": label_str}])], ignore_index=True)
    df.to_csv(label_file, index=False)
    print(f"Rótulo '{label_str}' salvo para {video}\n")

print("Todos os vídeos foram rotulados.")
