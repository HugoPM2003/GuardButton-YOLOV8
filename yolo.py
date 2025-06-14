import os
import cv2
import json
import numpy as np
import logging
from ultralytics import YOLO
from datetime import datetime
import joblib
from config import (
    OUTPUT_FRAMES_FOLDER, YOLO_RESULTS, FRAMESDECTECCAO,
    FRAME_SKIP, CONF_THRESHOLD, SAVE_FRAME_CONFIDENCE,
    MODEL_WEAPONS, MOVIMENTOS_CONFIRMADOS
)

# Configurações adicionais
ALERTA_CONF = 0.6  # Probabilidade mínima para alerta de movimento suspeito
DISTANCIA_MAX_ASSOCIACAO = 150  # Distância máxima para associar arma a pessoa
NORMALIZAR_KEYPOINTS = True  # Ativar normalização dos keypoints

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("detector_suspeitos.log"),
        logging.StreamHandler()
    ]
)

# Carrega modelos
model_weapon = YOLO(MODEL_WEAPONS)
model_pose = YOLO("yolov8n-pose.pt")
clf = joblib.load("modelo_roubo_keypoints_rf.joblib")

def normalizar_keypoints(flat_keypoints):
    """Normaliza keypoints para classificação (igual ao treino)."""
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

def keypoints_to_vector(keypoints):
    """Converte keypoints para vetor flat para o classificador."""
    flat = [coord for point in keypoints for coord in point]
    
    if NORMALIZAR_KEYPOINTS:
        flat = normalizar_keypoints(flat)
    
    return np.array(flat).reshape(1, -1)

def salvar_movimento_confirmado(frame, keypoints, arma_info, pessoa_idx, timestamp, video_name, frame_idx, prob_movimento=None):
    """Salva um movimento suspeito confirmado com imagem e metadados."""
    data_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{data_str}_{os.path.splitext(video_name)[0]}_f{frame_idx:05d}_pessoa{pessoa_idx}"
    
    if prob_movimento is not None:
        base_name += f"_mov{prob_movimento:.2f}"
    if arma_info is not None:
        base_name += f"_arma{arma_info['conf']:.2f}"

    # Salvar imagem
    os.makedirs(MOVIMENTOS_CONFIRMADOS, exist_ok=True)
    image_path = os.path.join(MOVIMENTOS_CONFIRMADOS, base_name + ".jpg")
    cv2.imwrite(image_path, frame)

    # Salvar vetor de keypoints
    vetor_path = os.path.join(MOVIMENTOS_CONFIRMADOS, base_name + ".npy")
    np.save(vetor_path, keypoints[pessoa_idx])

    # Salvar JSON com metadados
    data = {
        "timestamp_sec": timestamp,
        "frame": frame_idx,
        "video": video_name,
        "pessoa_index": pessoa_idx,
        "keypoints": keypoints[pessoa_idx],
        "arma_detectada": arma_info,
        "probabilidade_movimento_suspeito": prob_movimento
    }

    json_path = os.path.join(MOVIMENTOS_CONFIRMADOS, base_name + ".json")
    with open(json_path, "w") as jf:
        json.dump(data, jf, indent=2)

    logging.info(f"Movimento suspeito salvo: {base_name}")

def desenhar_deteccoes(frame, weapons, keypoints, alertas_movimento=None):
    """Desenha detecções no frame com cores diferenciadas."""
    COLOR_WEAPON = (0, 0, 255)    # Vermelho para armas
    COLOR_KEYPOINT = (0, 255, 0)  # Verde para keypoints
    COLOR_ALERTA = (0, 165, 255)  # Laranja para alertas de movimento
    COLOR_BOX_ALERTA = (0, 69, 255) # Vermelho escuro para bounding box de alerta

    # Desenha armas
    for w in weapons:
        x1, y1, x2, y2 = map(int, w['bbox'])
        conf = w['conf']
        label = f"Arma {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WEAPON, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WEAPON, 2)

    # Desenha keypoints e alertas
    for i, person in enumerate(keypoints):
        # Desenha keypoints
        for x, y in person:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 4, COLOR_KEYPOINT, -1)
        
        # Desenha alerta de movimento suspeito
        if alertas_movimento and i in alertas_movimento:
            prob = alertas_movimento[i]
            
            # Encontra bounding box aproximada da pessoa
            pontos_validos = np.array([(x, y) for x, y in person if x > 0 and y > 0])
            if len(pontos_validos) > 0:
                x1, y1 = pontos_validos.min(axis=0)
                x2, y2 = pontos_validos.max(axis=0)
                
                # Desenha bounding box
                cv2.rectangle(frame, (int(x1)-10, int(y1)-10), 
                             (int(x2)+10, int(y2)+10), COLOR_BOX_ALERTA, 2)
                
                # Desenha texto com probabilidade
                label = f"SUSPEITO: {prob:.2f}"
                cv2.putText(frame, label, (int(x1)-10, int(y1)-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ALERTA, 2)

    return frame

def process_video(video_path):
    """Processa um vídeo detectando armas, poses e movimentos suspeitos."""
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    frame_idx = 0
    evento_idx = 0
    event_log = []

    # Cria pastas de saída se não existirem
    os.makedirs(FRAMESDECTECCAO, exist_ok=True)
    os.makedirs(YOLO_RESULTS, exist_ok=True)
    os.makedirs(MOVIMENTOS_CONFIRMADOS, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inferência dos modelos
            results_weapon = model_weapon.predict(source=frame_rgb, conf=CONF_THRESHOLD, verbose=False)[0]
            results_pose = model_pose.predict(source=frame_rgb, conf=CONF_THRESHOLD, verbose=False)[0]

            # Processa detecções de armas
            weapons = []
            for box in results_weapon.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                weapons.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 2),
                    "class": int(cls)
                })

            # Processa keypoints das poses
            keypoints = []
            if results_pose.keypoints is not None:
                for person in results_pose.keypoints.xy:
                    keypoints.append(person.tolist())

            # Classifica movimentos suspeitos
            alertas_movimento = {}
            for i, person in enumerate(keypoints):
                if len(person) == 17:  # Verifica se tem todos os keypoints
                    try:
                        vec = keypoints_to_vector(person)
                        prob = clf.predict_proba(vec)[0][1]  # Probabilidade de ser suspeito
                        
                        if prob >= ALERTA_CONF:
                            alertas_movimento[i] = prob
                            logging.warning(
                                f"ALERTA: Movimento suspeito em {video_name}, "
                                f"frame {frame_idx}, pessoa {i} (prob={prob:.2f})"
                            )
                    except Exception as e:
                        logging.error(f"Erro ao classificar movimento: {e}")

            # Se houver armas ou movimentos suspeitos
            if weapons or alertas_movimento:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                arma_associacoes = []

                # Associa armas às pessoas mais próximas
                for arma in weapons:
                    x1, y1, x2, y2 = arma["bbox"]
                    centro_arma = ((x1 + x2) / 2, (y1 + y2) / 2)

                    menor_dist = float("inf")
                    pessoa_idx_associada = None

                    for idx, pontos in enumerate(keypoints):
                        pontos_validos = [(x, y) for x, y in pontos if x > 0 and y > 0]
                        if not pontos_validos:
                            continue
                            
                        # Calcula centro da pessoa
                        media_x = sum(x for x, _ in pontos_validos) / len(pontos_validos)
                        media_y = sum(y for _, y in pontos_validos) / len(pontos_validos)
                        dist = ((media_x - centro_arma[0]) ** 2 + (media_y - centro_arma[1]) ** 2) ** 0.5

                        if dist < menor_dist and dist < DISTANCIA_MAX_ASSOCIACAO:
                            menor_dist = dist
                            pessoa_idx_associada = idx

                    if pessoa_idx_associada is not None:
                        arma_associacoes.append({
                            "arma": arma,
                            "pessoa_index": pessoa_idx_associada,
                            "distancia": round(menor_dist, 2)
                        })

                # Prepara entrada do log
                log_entry = {
                    "frame": frame_idx,
                    "timestamp_sec": round(timestamp, 2),
                    "weapons_detected": weapons,
                    "keypoints": keypoints,
                    "arma_para_pessoa": arma_associacoes,
                    "alertas_movimento_suspeito": alertas_movimento
                }
                event_log.append(log_entry)

                # Verifica se deve salvar o frame
                max_conf_arma = max([w['conf'] for w in weapons]) if weapons else 0
                max_conf_movimento = max(alertas_movimento.values()) if alertas_movimento else 0

                if max_conf_arma >= SAVE_FRAME_CONFIDENCE or max_conf_movimento >= ALERTA_CONF:
                    # Desenha detecções no frame
                    desenhado = desenhar_deteccoes(frame.copy(), weapons, keypoints, alertas_movimento)
                    
                    # Define nome base para os arquivos
                    base_name = f"{os.path.splitext(video_name)[0]}_frame_{frame_idx:05d}"
                    if max_conf_arma > 0:
                        base_name += f"_arma_{max_conf_arma:.2f}"
                    if max_conf_movimento > 0:
                        base_name += f"_mov_{max_conf_movimento:.2f}"

                    # Salva frame com detecções
                    img_path = os.path.join(FRAMESDECTECCAO, base_name + ".jpg")
                    cv2.imwrite(img_path, desenhado)

                    # Salva metadados em JSON
                    meta_path = os.path.join(FRAMESDECTECCAO, base_name + "_keypoints.json")
                    with open(meta_path, "w") as jf:
                        json.dump(log_entry, jf, indent=2)

                    # Salva movimentos confirmados (armas associadas)
                    for assoc in arma_associacoes:
                        salvar_movimento_confirmado(
                            frame=desenhado,
                            keypoints=keypoints,
                            arma_info=assoc['arma'],
                            pessoa_idx=assoc['pessoa_index'],
                            timestamp=timestamp,
                            video_name=video_name,
                            frame_idx=frame_idx
                        )

                    # Salva movimentos suspeitos sem arma associada
                    for pessoa_idx, prob in alertas_movimento.items():
                        if not any(a['pessoa_index'] == pessoa_idx for a in arma_associacoes):
                            salvar_movimento_confirmado(
                                frame=desenhado,
                                keypoints=keypoints,
                                arma_info=None,
                                pessoa_idx=pessoa_idx,
                                timestamp=timestamp,
                                video_name=video_name,
                                frame_idx=frame_idx,
                                prob_movimento=prob
                            )

                    evento_idx += 1

        frame_idx += 1

    cap.release()

    # Salva log final do vídeo
    log_path = os.path.join(YOLO_RESULTS, f"{os.path.splitext(video_name)[0]}_log.json")
    with open(log_path, "w") as f:
        json.dump(event_log, f, indent=2)

    logging.info(
        f"Processamento concluído para {video_name}. "
        f"Total de frames: {frame_idx}, eventos registrados: {evento_idx}"
    )

# Processa todos os vídeos na pasta de entrada
if __name__ == "__main__":
    logging.info("Iniciando processamento de vídeos...")
    
    for file in sorted(os.listdir(OUTPUT_FRAMES_FOLDER)):
        if file.endswith(".avi"):
            video_path = os.path.join(OUTPUT_FRAMES_FOLDER, file)
            logging.info(f"Processando vídeo: {file}")
            try:
                process_video(video_path)
            except Exception as e:
                logging.error(f"Erro ao processar {file}: {str(e)}")
    
    logging.info("Processamento de todos os vídeos concluído.")