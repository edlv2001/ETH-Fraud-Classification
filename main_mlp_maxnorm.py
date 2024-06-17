import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import Sequential, layers
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.metrics import Precision, Recall


# Definir el compilador del modelo MLP
def compile_mlp(input_dim, H, num_epochs, num_layers, activation, dropout_probability):
    # Crear un MLP secuencial
    model_n = Sequential()

    model_n.add(layers.Dense(H, input_shape=(input_dim,), activation=activation))

    for _ in range(num_layers - 1):
        model_n.add(layers.Dense(H, activation=activation, kernel_constraint=MaxNorm(3)))
        model_n.add(layers.Dropout(dropout_probability))

    model_n.add(layers.Dense(1, activation='sigmoid'))

    # Configurar el modelo
    model_n.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Precision(), Recall()])

    return model_n


# Cargar el dataset
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

# Obtener las dimensiones de entrada
input_dim = X_train_resampled.shape[1]

# Inicializar y compilar el modelo MLP
H = 100  # Tamaño de las capas ocultas
num_epochs = 50
num_layers = 3
activation = 'relu'
dropout_probability = 0.5

mlp_model = compile_mlp(input_dim, H, num_epochs, num_layers, activation, dropout_probability)

# Entrenar el modelo
mlp_model.fit(X_train_resampled, y_train_resampled, epochs=num_epochs, batch_size=32, validation_split=0.2)

# Predecir en el conjunto de prueba
y_pred_proba = mlp_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")

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
