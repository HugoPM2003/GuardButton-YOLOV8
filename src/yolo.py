# Importa todas as bibliotecas necessárias para o programa funcionar
import os          # Para trabalhar com pastas e arquivos
import cv2         # Biblioteca para processamento de vídeos e imagens (OpenCV)
import json        # Para salvar informações em formato JSON (um tipo de arquivo de dados)
import numpy as np # Para cálculos matemáticos complexos
import logging     # Para registrar o que está acontecendo durante a execução
from ultralytics import YOLO  # Modelo de inteligência artificial para detectar objetos e poses
from datetime import datetime # Para trabalhar com datas e horários
import joblib      # Para carregar modelos de machine learning salvos

# Importa configurações do arquivo config.py
from config import (
    OUTPUT_FRAMES_FOLDER,  # Pasta onde estão os vídeos para analisar
    YOLO_RESULTS,         # Pasta para salvar os resultados
    FRAMESDECTECCAO,      # Pasta para salvar frames com detecções
    FRAME_SKIP,           # Quantos frames pular entre análises (para ser mais rápido)
    CONF_THRESHOLD,       # Nível mínimo de confiança para considerar uma detecção válida
    SAVE_FRAME_CONFIDENCE, # Nível de confiança para salvar um frame com detecção
    MODEL_WEAPONS,        # Arquivo do modelo que detecta armas
    MOVIMENTOS_CONFIRMADOS # Pasta para salvar movimentos suspeitos confirmados
)

# ==============================================
# CONFIGURAÇÕES ADICIONAIS (AJUSTÁVEIS)
# ==============================================
ALERTA_CONF = 0.6  # Probabilidade mínima para considerar um movimento como suspeito
DISTANCIA_MAX_ASSOCIACAO = 150  # Distância máxima para associar uma arma a uma pessoa
NORMALIZAR_KEYPOINTS = True  # Ativa a normalização dos pontos do corpo (keypoints)

# ==============================================
# CONFIGURA O SISTEMA DE LOG (REGISTRO DE EVENTOS)
# ==============================================
# Isso vai criar um arquivo de log e também mostrar mensagens na tela
logging.basicConfig(
    level=logging.INFO,  # Nível de detalhe das mensagens
    format="%(asctime)s [%(levelname)s] %(message)s",  # Formato das mensagens
    handlers=[
        logging.FileHandler("detector_suspeitos.log"),  # Arquivo de log
        logging.StreamHandler()  # Mostrar também na tela
    ]
)

# ==============================================
# CARREGA OS MODELOS DE INTELIGÊNCIA ARTIFICIAL
# ==============================================
model_weapon = YOLO(MODEL_WEAPONS)  # Modelo para detectar arma e faca 
model_pose = YOLO("yolov8n-pose.pt")  # Modelo para detectar poses humanas(ja disponibilizado pela yolo)
clf = joblib.load("modelo_roubo_keypoints_rf.joblib")  # Modelo para classificar movimentos suspeitos (precisa treinar com o treinar_classificador.py)

# ==============================================
# FUNÇÃO: normalizar_keypoints
# ==============================================
def normalizar_keypoints(flat_keypoints):
    """
    Ajusta os pontos do corpo (keypoints) para um tamanho padrão,
    igual foi feito durante o treinamento do modelo.
    Isso ajuda a classificar os movimentos corretamente.
    """
    # Converte os pontos para um formato que o computador entende melhor
    coords = np.array(flat_keypoints).reshape(-1, 2)
    # Filtra apenas pontos válidos (diferentes de zero)
    valid_coords = coords[(coords != 0).all(axis=1)]

    # Se não tiver pontos válidos, retorna os originais
    if len(valid_coords) == 0:
        return flat_keypoints

    # Calcula o centro do corpo
    center = valid_coords.mean(axis=0)
    coords -= center  # Centraliza os pontos

    # Calcula o tamanho do corpo e ajusta a escala
    scale = np.linalg.norm(valid_coords.max(axis=0) - valid_coords.min(axis=0))
    if scale > 0:
        coords /= scale  # Redimensiona os pontos

    # Retorna os pontos normalizados no formato original
    return coords.flatten().tolist()

# ==============================================
# FUNÇÃO: keypoints_to_vector
# ==============================================
def keypoints_to_vector(keypoints):
    """
    Converte os pontos do corpo (keypoints) para um formato que
    o modelo de classificação de movimentos consegue entender.
    """
    # Achata a lista de pontos (transforma em uma linha só)
    flat = [coord for point in keypoints for coord in point]
    
    # Se a normalização estiver ativada, aplica a função de normalização
    if NORMALIZAR_KEYPOINTS:
        flat = normalizar_keypoints(flat)
    
    # Retorna no formato que o modelo espera
    return np.array(flat).reshape(1, -1)

# ==============================================
# FUNÇÃO: salvar_movimento_confirmado
# ==============================================
def salvar_movimento_confirmado(frame, keypoints, arma_info, pessoa_idx, timestamp, video_name, frame_idx, prob_movimento=None):
    """
    Salva um movimento suspeito confirmado, incluindo:
    - A imagem do frame
    - Os pontos do corpo (keypoints)
    - Informações sobre armas detectadas
    - Metadados (dados sobre os dados)
    """
    # Cria um nome único para os arquivos baseado na data, nome do vídeo e frame
    data_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{data_str}_{os.path.splitext(video_name)[0]}_f{frame_idx:05d}_pessoa{pessoa_idx}"
    
    # Adiciona informações sobre movimento e armas ao nome do arquivo
    if prob_movimento is not None:
        base_name += f"_mov{prob_movimento:.2f}"
    if arma_info is not None:
        base_name += f"_arma{arma_info['conf']:.2f}"

    # Cria a pasta se não existir e salva a imagem
    os.makedirs(MOVIMENTOS_CONFIRMADOS, exist_ok=True)
    image_path = os.path.join(MOVIMENTOS_CONFIRMADOS, base_name + ".jpg")
    cv2.imwrite(image_path, frame)

    # Salva os pontos do corpo em um arquivo separado
    vetor_path = os.path.join(MOVIMENTOS_CONFIRMADOS, base_name + ".npy")
    np.save(vetor_path, keypoints[pessoa_idx])

    # Prepara todos os dados para salvar em JSON
    data = {
        "timestamp_sec": timestamp,  # Momento no vídeo em segundos
        "frame": frame_idx,         # Número do frame no vídeo
        "video": video_name,        # Nome do vídeo original
        "pessoa_index": pessoa_idx, # Índice da pessoa detectada
        "keypoints": keypoints[pessoa_idx],  # Pontos do corpo
        "arma_detectada": arma_info,  # Informações sobre arma (se houver)
        "probabilidade_movimento_suspeito": prob_movimento  # Chance de ser suspeito
    }

    # Salva os metadados em JSON
    json_path = os.path.join(MOVIMENTOS_CONFIRMADOS, base_name + ".json")
    with open(json_path, "w") as jf:
        json.dump(data, jf, indent=2)

    # Registra no log que salvou um movimento suspeito
    logging.info(f"Movimento suspeito salvo: {base_name}")

# ==============================================
# FUNÇÃO: desenhar_deteccoes
# ==============================================
def desenhar_deteccoes(frame, weapons, keypoints, alertas_movimento=None):
    """
    Desenha na imagem as detecções feitas pelos modelos:
    - Caixas ao redor de armas (vermelho)
    - Pontos do corpo (verde)
    - Alertas de movimento suspeito (laranja/vermelho)
    """
    # Define cores para cada tipo de detecção
    COLOR_WEAPON = (0, 0, 255)    # Vermelho para armas
    COLOR_KEYPOINT = (0, 255, 0)  # Verde para pontos do corpo
    COLOR_ALERTA = (0, 165, 255)  # Laranja para alertas de movimento
    COLOR_BOX_ALERTA = (0, 69, 255) # Vermelho escuro para caixa de alerta

    # Desenha cada arma detectada
    for w in weapons:
        x1, y1, x2, y2 = map(int, w['bbox'])  # Coordenadas da caixa ao redor da arma
        conf = w['conf']  # Nível de confiança da detecção
        label = f"Arma {conf:.2f}"  # Texto para mostrar na imagem
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WEAPON, 2)  # Desenha caixa
        cv2.putText(frame, label, (x1, y1 - 10),  # Escreve texto acima da caixa
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WEAPON, 2)

    # Desenha os pontos do corpo e alertas de movimento
    for i, person in enumerate(keypoints):
        # Desenha cada ponto do corpo (se for válido)
        for x, y in person:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 4, COLOR_KEYPOINT, -1)
        
        # Se houver alerta de movimento suspeito para esta pessoa
        if alertas_movimento and i in alertas_movimento:
            prob = alertas_movimento[i]  # Probabilidade de ser suspeito
            
            # Encontra uma caixa ao redor de todos os pontos do corpo
            pontos_validos = np.array([(x, y) for x, y in person if x > 0 and y > 0])
            if len(pontos_validos) > 0:
                x1, y1 = pontos_validos.min(axis=0)  # Canto superior esquerdo
                x2, y2 = pontos_validos.max(axis=0)  # Canto inferior direito
                
                # Desenha a caixa ao redor da pessoa
                cv2.rectangle(frame, (int(x1)-10, int(y1)-10), 
                             (int(x2)+10, int(y2)+10), COLOR_BOX_ALERTA, 2)
                
                # Escreve o texto de alerta
                label = f"SUSPEITO: {prob:.2f}"
                cv2.putText(frame, label, (int(x1)-10, int(y1)-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ALERTA, 2)

    return frame  # Retorna a imagem com as detecções desenhadas

# ==============================================
# FUNÇÃO PRINCIPAL: process_video
# ==============================================
def process_video(video_path):
    """
    Função principal que processa um vídeo completo, analisando:
    - Detecção de armas
    - Detecção de poses humanas
    - Classificação de movimentos suspeitos
    - Associação de armas a pessoas
    - Salvamento de eventos suspeitos
    """
    # Abre o vídeo para leitura
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)  # Pega só o nome do arquivo
    frame_idx = 0  # Contador de frames
    evento_idx = 0  # Contador de eventos suspeitos
    event_log = []  # Lista para guardar todos os eventos detectados

    # Cria as pastas de saída se não existirem
    os.makedirs(FRAMESDECTECCAO, exist_ok=True)
    os.makedirs(YOLO_RESULTS, exist_ok=True)
    os.makedirs(MOVIMENTOS_CONFIRMADOS, exist_ok=True)

    # Loop principal: processa cada frame do vídeo
    while cap.isOpened():
        ret, frame = cap.read()  # Lê o próximo frame
        if not ret:  # Se não conseguir ler, acabou o vídeo
            break

        # Processa apenas 1 frame a cada FRAME_SKIP (para ser mais rápido)
        if frame_idx % FRAME_SKIP == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte cores

            # Usa o modelo YOLO para detectar armas no frame
            results_weapon = model_weapon.predict(source=frame_rgb, conf=CONF_THRESHOLD, verbose=False)[0]
            # Usa outro modelo YOLO para detectar poses humanas
            results_pose = model_pose.predict(source=frame_rgb, conf=CONF_THRESHOLD, verbose=False)[0]

            # Processa as armas detectadas
            weapons = []
            for box in results_weapon.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box  # Extrai informações da detecção
                weapons.append({
                    "bbox": [x1, y1, x2, y2],  # Coordenadas da caixa
                    "conf": round(conf, 2),     # Confiança da detecção (0-1)
                    "class": int(cls)           # Classe do objeto detectado
                })

            # Processa os pontos do corpo (keypoints) das pessoas detectadas
            keypoints = []
            if results_pose.keypoints is not None:
                for person in results_pose.keypoints.xy:  # Para cada pessoa
                    keypoints.append(person.tolist())  # Converte para lista

            # Classifica os movimentos como suspeitos ou não
            alertas_movimento = {}
            for i, person in enumerate(keypoints):
                if len(person) == 17:  # Verifica se detectou todos os pontos do corpo
                    try:
                        # Prepara os pontos para o classificador
                        vec = keypoints_to_vector(person)
                        # Pede ao modelo para classificar (probabilidade de ser suspeito)
                        prob = clf.predict_proba(vec)[0][1]
                        
                        # Se a probabilidade for alta o suficiente, marca como alerta
                        if prob >= ALERTA_CONF:
                            alertas_movimento[i] = prob
                            logging.warning(
                                f"ALERTA: Movimento suspeito em {video_name}, "
                                f"frame {frame_idx}, pessoa {i} (prob={prob:.2f})"
                            )
                    except Exception as e:
                        logging.error(f"Erro ao classificar movimento: {e}")

            # Se detectou armas OU movimentos suspeitos neste frame
            if weapons or alertas_movimento:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Tempo no vídeo
                arma_associacoes = []  # Para guardar associações de armas a pessoas

                # Tenta associar cada arma detectada à pessoa mais próxima
                for arma in weapons:
                    x1, y1, x2, y2 = arma["bbox"]
                    centro_arma = ((x1 + x2) / 2, (y1 + y2) / 2)  # Centro da arma

                    menor_dist = float("inf")  # Inicia com distância muito grande
                    pessoa_idx_associada = None  # Ainda não associou a ninguém

                    # Para cada pessoa detectada
                    for idx, pontos in enumerate(keypoints):
                        # Filtra apenas pontos válidos do corpo
                        pontos_validos = [(x, y) for x, y in pontos if x > 0 and y > 0]
                        if not pontos_validos:
                            continue
                            
                        # Calcula o centro do corpo da pessoa
                        media_x = sum(x for x, _ in pontos_validos) / len(pontos_validos)
                        media_y = sum(y for _, y in pontos_validos) / len(pontos_validos)
                        # Calcula distância entre a arma e a pessoa
                        dist = ((media_x - centro_arma[0]) ** 2 + (media_y - centro_arma[1]) ** 2) ** 0.5

                        # Se for a pessoa mais próxima até agora e dentro do limite
                        if dist < menor_dist and dist < DISTANCIA_MAX_ASSOCIACAO:
                            menor_dist = dist
                            pessoa_idx_associada = idx

                    # Se encontrou uma pessoa próxima o suficiente, guarda a associação
                    if pessoa_idx_associada is not None:
                        arma_associacoes.append({
                            "arma": arma,
                            "pessoa_index": pessoa_idx_associada,
                            "distancia": round(menor_dist, 2)
                        })

                # Prepara os dados deste evento para o log
                log_entry = {
                    "frame": frame_idx,
                    "timestamp_sec": round(timestamp, 2),
                    "weapons_detected": weapons,
                    "keypoints": keypoints,
                    "arma_para_pessoa": arma_associacoes,
                    "alertas_movimento_suspeito": alertas_movimento
                }
                event_log.append(log_entry)

                # Verifica se a detecção é importante o suficiente para salvar
                max_conf_arma = max([w['conf'] for w in weapons]) if weapons else 0
                max_conf_movimento = max(alertas_movimento.values()) if alertas_movimento else 0

                # Se passar dos limiares de confiança configurados
                if max_conf_arma >= SAVE_FRAME_CONFIDENCE or max_conf_movimento >= ALERTA_CONF:
                    # Desenha todas as detecções no frame
                    desenhado = desenhar_deteccoes(frame.copy(), weapons, keypoints, alertas_movimento)
                    
                    # Cria um nome base para os arquivos
                    base_name = f"{os.path.splitext(video_name)[0]}_frame_{frame_idx:05d}"
                    if max_conf_arma > 0:
                        base_name += f"_arma_{max_conf_arma:.2f}"
                    if max_conf_movimento > 0:
                        base_name += f"_mov_{max_conf_movimento:.2f}"

                    # Salva a imagem com as detecções desenhadas
                    img_path = os.path.join(FRAMESDECTECCAO, base_name + ".jpg")
                    cv2.imwrite(img_path, desenhado)

                    # Salva os metadados em JSON
                    meta_path = os.path.join(FRAMESDECTECCAO, base_name + "_keypoints.json")
                    with open(meta_path, "w") as jf:
                        json.dump(log_entry, jf, indent=2)

                    # Salva movimentos confirmados onde armas foram associadas a pessoas
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

                    evento_idx += 1  # Incrementa contador de eventos

        frame_idx += 1  # Vai para o próximo frame

    cap.release()  # Fecha o vídeo quando terminar

    # Salva o log completo com todos os eventos detectados no vídeo
    log_path = os.path.join(YOLO_RESULTS, f"{os.path.splitext(video_name)[0]}_log.json")
    with open(log_path, "w") as f:
        json.dump(event_log, f, indent=2)

    logging.info(
        f"Processamento concluído para {video_name}. "
        f"Total de frames: {frame_idx}, eventos registrados: {evento_idx}"
    )

# ==============================================
# EXECUÇÃO PRINCIPAL (QUANDO O SCRIPT É CHAMADO)
# ==============================================
if __name__ == "__main__":
    logging.info("Iniciando processamento de vídeos...")
    
    # Para cada arquivo na pasta de vídeos
    for file in sorted(os.listdir(OUTPUT_FRAMES_FOLDER)):
        if file.endswith(".avi"):  # Se for um vídeo AVI
            video_path = os.path.join(OUTPUT_FRAMES_FOLDER, file)
            logging.info(f"Processando vídeo: {file}")
            try:
                process_video(video_path)  # Processa o vídeo
            except Exception as e:
                logging.error(f"Erro ao processar {file}: {str(e)}")
    
    logging.info("Processamento de todos os vídeos concluído.")
