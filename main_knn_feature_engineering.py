import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pycm import ConfusionMatrix, Compare
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FOLD_NUMBER = 10
RANDOM_STATE = 23
METRIC_LIST = ["Accuracy", "F1", "Kappa", "Precision", "Recall"]
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
df = pd.read_csv("data/Merged_Dataset.csv")

# Separar características y etiqueta
X = df.drop(columns=['Address', 'Flag'])
y = df['Flag']

# Calcular la matriz de correlación
correlation_matrix = X.corr().abs()

# Identificar las características altamente correlacionadas
high_corr_var = np.where(correlation_matrix > 0.75)
high_corr_var = [(correlation_matrix.index[x], correlation_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]

# Crear nuevas características combinando las altamente correlacionadas
for var1, var2 in high_corr_var:
    # X[f'{var1}_x_{var2}'] = X[var1] * X[var2]
    X[f'{var1}_plus_{var2}'] = X[var1] + X[var2]

# Crear un conjunto para las variables a eliminar
variables_a_eliminar = set()
for var1, var2 in high_corr_var:
    variables_a_eliminar.add(var1)
    variables_a_eliminar.add(var2)

# Eliminar las variables originales altamente correlacionadas
X = X.drop(columns=list(variables_a_eliminar))

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE para equilibrar las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Rellenar valores faltantes con la media de cada columna en el conjunto de entrenamiento y prueba
X_train_resampled = X_train_resampled.fillna(X_train_resampled.mean())
X_test = X_test.fillna(X_test.mean())

# Inicializar el modelo KNN
knn_params = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto'
}
knn_model = KNeighborsClassifier(**knn_params)

# Estandarizar los datos
scaler = StandardScaler().fit(X_train_resampled)
X_train_resampled = scaler.transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Entrenar el modelo
knn_model.fit(X_train_resampled, y_train_resampled)

# Predecir en el conjunto de prueba
y_pred = knn_model.predict(X_test)

# Calcular las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mostrar las métricas
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
