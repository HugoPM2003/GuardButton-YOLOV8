import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Carrega dados
X = np.load("X.npy")
y = np.load("y.npy")

# Exemplo: aqui só temos exemplos de roubo (y=1)
# Então precisamos de exemplos negativos (y=0) para treinar. 
# Se não tiver, o classificador não vai aprender.
# Para efeito didático, vou criar dados negativos aleatórios (melhor coletar reais!).

import numpy.random as npr

num_negativos = len(y)
X_neg = npr.uniform(low=0, high=1, size=(num_negativos, X.shape[1]))
y_neg = npr.zeros(num_negativos, dtype=int)

# Concatena positivos e negativos
X_all = np.vstack((X, X_neg))
y_all = np.concatenate((y, y_neg))

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Treina RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Avalia
y_pred = clf.predict(X_test)

print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

print("[INFO] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Salva modelo (opcional)
import joblib
joblib.dump(clf, "modelo_roubo_keypoints_rf.joblib")
print("[✓] Modelo salvo: modelo_roubo_keypoints_rf.joblib")
