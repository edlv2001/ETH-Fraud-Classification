import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
df = pd.read_csv("../data/Merged_Dataset.csv")

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

# Inicializar el modelo de regresión logística
logistic_params = {
    'random_state': 42,
    'max_iter': 300
}
logistic_model = LogisticRegression(**logistic_params)

# Estandarizar los datos
scaler = StandardScaler().fit(X_train_resampled)
X_train_resampled = scaler.transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Entrenar el modelo
logistic_model.fit(X_train_resampled, y_train_resampled)

# Predecir en el conjunto de prueba
y_pred = logistic_model.predict(X_test)

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
