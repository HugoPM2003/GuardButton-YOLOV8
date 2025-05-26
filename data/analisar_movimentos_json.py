import os
import json
import pandas as pd

PASTA_POSES = "../poses"
ARQUIVO_LABELS = "labels.csv"

def detectar_movimento_roubo_json(dados):
    try:
        for i in range(1, len(dados)):
            atual = dados[i]
            anterior = dados[i - 1]

            if not atual or not anterior:
                continue

            kp_atual = atual.get("keypoints", [])
            kp_anterior = anterior.get("keypoints", [])

            # Verifica se há keypoints suficientes (índice 1 = nose, 4 = right_wrist)
            if len(kp_atual) <= 4 or len(kp_anterior) <= 4:
                continue

            nose_y_atual = kp_atual[1][1]
            wrist_y_atual = kp_atual[4][1]
            nose_y_ant = kp_anterior[1][1]
            wrist_y_ant = kp_anterior[4][1]

            if wrist_y_atual < nose_y_atual and wrist_y_ant > nose_y_ant:
                return "roubo"

        return "normal"
    except Exception as e:
        print("⚠️ Erro durante análise:", e)
        return "erro"


# Lê o arquivo CSV original
labels_df = pd.read_csv(ARQUIVO_LABELS)

resultados = []

for index, row in labels_df.iterrows():
    nome_video = row["video"]
    nome_json = nome_video.replace(".mp4", "")
    json_match = [f for f in os.listdir(PASTA_POSES) if f.startswith(nome_json) and f.endswith(".json")]

    if json_match:
        caminho = os.path.join(PASTA_POSES, json_match[0])
        with open(caminho, "r", encoding="utf-8") as f:
            dados_json = json.load(f)

        classificacao = detectar_movimento_roubo_json(dados_json)
        resultados.append([nome_video, classificacao])
        print(f"✅ {nome_video}: {classificacao}")
    else:
        print(f"❌ Arquivo JSON não encontrado para {nome_video}")
        resultados.append([nome_video, "nao_encontrado"])

# Salvar resultados
pd.DataFrame(resultados, columns=["video", "classificacao"]).to_csv("labels_atualizado.csv", index=False)
print("✅ Arquivo labels_atualizado.csv criado com sucesso!")
