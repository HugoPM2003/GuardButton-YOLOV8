import os
import json
import numpy as np

# Diretórios de onde extrair os exemplos positivos
BASE_DIRS = [
    "movimentos_confirmados/arma_detectada",
    "movimentos_confirmados/confirmados_pelo_segurança"
]

X = []
y = []

def carregar_keypoints_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    keypoints = data["keypoints"]

    # Flatten keypoints: [[x1,y1], [x2,y2], ...] → [x1, y1, x2, y2, ..., x17, y17]
    flat = [coord for point in keypoints for coord in point]
    return flat

# Coleta os exemplos positivos
for folder in BASE_DIRS:
    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        if file.endswith(".json"):
            path = os.path.join(folder, file)
            try:
                vetor = carregar_keypoints_json(path)
                if len(vetor) == 34:  # 17 keypoints x 2 (x, y)
                    X.append(vetor)
                    y.append(1)
                else:
                    print(f"[!] Ignorado (keypoints incompletos): {file}")
            except Exception as e:
                print(f"[!] Erro ao ler {file}: {e}")
                
def normalizar_keypoints(flat_keypoints):
    coords = np.array(flat_keypoints).reshape(-1, 2)
    valid_coords = coords[(coords != 0).all(axis=1)]  # ignora pontos zerados

    if len(valid_coords) == 0:
        return flat_keypoints  # se tudo for 0, retorna original

    center = valid_coords.mean(axis=0)
    coords -= center

    scale = np.linalg.norm(valid_coords.max(axis=0) - valid_coords.min(axis=0))
    if scale > 0:
        coords /= scale

    return coords.flatten().tolist()


# Converte para np.array
X = np.array(X)
y = np.array(y)

# Salva os vetores
np.save("X.npy", X)
np.save("y.npy", y)

print(f"[✓] Vetores salvos — total exemplos: {len(y)}")
