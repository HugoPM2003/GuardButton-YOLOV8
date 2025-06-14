import os

# Configurações de diretórios (use caminhos absolutos para melhor compatibilidade)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Pega o diretório do config.py

# Pasta onde os vídeos de entrada estão armazenados
VIDEO_FOLDER = os.path.join(BASE_DIR, r"C:\Users\Bruno\AppData\Roaming\Security Eye\snapshots_video")

# Pasta onde os frames extraídos serão salvos
OUTPUT_FRAMES_FOLDER = os.path.join(BASE_DIR, r"C:\Users\Bruno\Desktop\smart_security\src\framesegmentados")

# Configurações opcionais de processamento
FRAME_SIZE = (640, 640)  # Largura, Altura

YOLO_RESULTS =  os.path.join(BASE_DIR, r"C:\Users\Bruno\Desktop\smart_security\src\resultados")

FRAMESDECTECCAO = os.path.join(BASE_DIR, r"C:\Users\Bruno\Desktop\smart_security\src\resultados_frames")

FRAME_SKIP = 5

CONF_THRESHOLD = 0.3

SAVE_FRAME_CONFIDENCE = 0.75

MODEL_WEAPONS = r"C:\Users\Bruno\Desktop\smart_security\models\yolo\yolo_furtos_v1\weights\best.pt"

MOVIMENTOS_CONFIRMADOS = r"C:\Users\Bruno\Desktop\smart_security\src\movimentos_confirmados"

OUTPUT_FOLDER = r"C:\Users\Bruno\Desktop\smart_security\src\movimentos_confirmados\confirmados_pelo_segurança"