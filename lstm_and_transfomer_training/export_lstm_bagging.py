import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Definir el tamaño de la secuencia (asegúrate de que coincida con el tamaño usado al generar las secuencias)
sequence_length = 40

# Cargar las secuencias serializadas
with open('../data/sequences.pkl', 'rb') as f:
    X, y = pickle.load(f)

with open('../data/sequences_2.pkl', 'rb') as f:
    X2, y2 = pickle.load(f)

X = np.concatenate((X, X2), axis= 0)

y = np.concatenate((y, y2), axis=0)


if sequence_length > len(X[0]):
    sequence_length = len(X[0])
else:
    X = np.array([seq[:sequence_length] for seq in X])

print("Número de datos")
print(len(y))

with open('../data/column_names.pkl', 'rb') as f:
    columns = pickle.load(f)

# Identificar las columnas numéricas y las columnas de acción
num_columns = [
    'blockNumber', 'timeStamp', 'nonce', 'gas', 'gasPrice',
    'isError', 'txreceipt_status', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'value'
]
action_columns = ['action_made', 'action_received']

print(len(X[0][0]))
print(len(columns))
print(columns)

# Separar las columnas numéricas y de acción en X
X_num = X[:, :, [columns.index(col) for col in num_columns]]
X_action = X[:, :, [columns.index(col) for col in action_columns]]

# Imputar valores NaN en las columnas numéricas
imputer = SimpleImputer(strategy='mean')
X_num_reshaped = X_num.reshape(-1, len(num_columns))
X_num_imputed = imputer.fit_transform(X_num_reshaped)
X_num = X_num_imputed.reshape(-1, sequence_length, len(num_columns))

print("Primeras 5 filas de X:")
print(X_num[:5])

print("Primeras 5 filas de y:")
print(y[:5])

print("Número de datos")
print(len(y))

# Escalar las características numéricas
scaler = MinMaxScaler()
X_num_reshaped = X_num.reshape(-1, len(num_columns))
X_num_scaled = scaler.fit_transform(X_num_reshaped)
X_num = X_num_scaled.reshape(-1, sequence_length, len(num_columns))

# Exportar los modelos con pickle
with open('../exported_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Concatenar las características numéricas escaladas con las columnas de acción
X_combined = np.concatenate([X_num, X_action], axis=2)

# Dividir los datos en entrenamiento y prueba antes de aplicar SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE solo al conjunto de entrenamiento
X_train_reshaped = X_train.reshape(-1, (len(num_columns) + len(action_columns)) * sequence_length)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_reshaped, y_train)

# Restaurar la forma de X_train_res
X_train_res = X_train_res.reshape(-1, sequence_length, len(num_columns) + len(action_columns))

# Dividir el conjunto de entrenamiento en entrenamiento y validación
X_train_res, X_val, y_train_res, y_val = train_test_split(X_train_res, y_train_res, test_size=0.2, random_state=42, stratify=y_train_res)

# Definir una función para crear el modelo LSTM
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(Dense(10, activation='softmax'))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

n_estimators = 10
models = []
predictions = []
input_shape = (sequence_length, X_combined.shape[-1])

for i in range(n_estimators):
    print("\n\n\nMODELO ", i,"\n\n")
    model = create_lstm_model(input_shape)

    # Definir los callbacks para EarlyStopping y ModelCheckpoint
    checkpoint_filepath = f'best_model_{i}.weights.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

    # Entrenar el modelo con los callbacks
    model.fit(X_train_res, y_train_res, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1,
              callbacks=[model_checkpoint_callback, early_stopping_callback])

    # Cargar los pesos del mejor modelo guardado
    model.load_weights(checkpoint_filepath)

    # Verificar los datos antes de la predicción
    X_test = np.nan_to_num(X_test)
    if not np.issubdtype(X_test.dtype, np.float32):
        X_test = X_test.astype(np.float32)

    y_pred = model.predict(X_test)
    models.append(model)
    predictions.append(y_pred)

# Promediar las predicciones
y_pred_avg = np.mean(predictions, axis=0)
y_pred_avg = (y_pred_avg > 0.5).astype("int32")

# Exportar los modelos con pickle
with open('../exported_models/lstm_ensemble_models.pkl', 'wb') as f:
    pickle.dump(models, f)

# Calcular las métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred_avg)
precision = precision_score(y_test, y_pred_avg)
recall = recall_score(y_test, y_pred_avg)
f1 = f1_score(y_test, y_pred_avg)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_avg)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
