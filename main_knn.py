import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE para equilibrar las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Rellenar valores faltantes con la media de cada columna en el conjunto de entrenamiento y prueba
X_train_resampled = X_train_resampled.fillna(X_train_resampled.mean())
X_test = X_test.fillna(X_test.mean())

# Estandarizar los datos
scaler = StandardScaler().fit(X_train_resampled)
X_train_resampled = scaler.transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Inicializar el modelo KNN
knn_model = KNeighborsClassifier()

# Definir el rango de valores de k a probar
param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto']
}

# Configurar el GridSearchCV
grid_search = GridSearchCV(knn_model, param_grid, cv=FOLD_NUMBER, scoring='accuracy', n_jobs=-1)

# Entrenar el modelo con GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Obtener el mejor modelo y sus parámetros
best_knn_model = grid_search.best_estimator_
print(f"Best K: {grid_search.best_params_['n_neighbors']}")
print(f"Best Parameters: {grid_search.best_params_}")

# Predecir en el conjunto de prueba con el mejor modelo encontrado
y_pred = best_knn_model.predict(X_test)

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
