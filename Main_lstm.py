import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

API_KEY = os.environ.get('ETHERSCAN_API_KEY')

dtype_spec = {
    'blockNumber': 'object',
    'timeStamp': np.int64,
    'nonce': np.int64,
    'gas': np.int64,
    'gasPrice': 'object',
    'isError': np.int64,
    'txreceipt_status': np.float64,
    'cumulativeGasUsed': np.int64,
    'gasUsed': np.int64,
    'confirmations': np.int64,
    'value': 'object'
}

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Función para obtener las transacciones de una cuenta
def get_transactions_by_address(address, start_block=0, end_block=99999999, sort='desc', max_length=None):
    url = f'https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock={start_block}&endblock={end_block}&sort={sort}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    if data['status'] == '1':
        transactions = pd.DataFrame(data['result'])
        # Ordenar transacciones por tiempo de creación, de más reciente a más antigua
        transactions['timeStamp'] = transactions['timeStamp'].astype(int)
        transactions = transactions.sort_values(by='timeStamp', ascending=False)
        # Limitar el número de transacciones si se especifica max_length
        if max_length is not None:
            transactions = transactions.head(max_length)
        return transactions
    else:
        print('Error:', data['message'])
        return pd.DataFrame()

# Función para crear el dataset con secuencias de transacciones previas
def create_sequence_dataset(transactions, address, sequence_length, scaler):
    # Añadir columnas action_made y action_received
    transactions['action_made'] = np.where(transactions['from'].str.lower() == address.lower(), 1, 0)
    transactions['action_received'] = np.where(transactions['to'].str.lower() == address.lower(), 1, 0)

    # Mantener solo las columnas especificadas
    columns_to_keep = [
        'blockNumber', 'timeStamp', 'nonce', 'gas', 'gasPrice',
        'isError', 'txreceipt_status', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'value',
        'action_made', 'action_received'
    ]
    transactions = transactions[columns_to_keep].copy()

    # Convertir los tipos de datos usando loc para evitar SettingWithCopyWarning
    transactions.loc[:, 'txreceipt_status'] = pd.to_numeric(transactions['txreceipt_status'], errors='coerce')
    transactions.loc[:, 'value'] = pd.to_numeric(transactions['value'], errors='coerce')

    transactions = transactions.astype(dtype_spec)

    # Escalar las columnas numéricas
    numeric_columns = [
        'blockNumber', 'timeStamp', 'nonce', 'gas', 'gasPrice',
        'isError', 'txreceipt_status', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'value'
    ]

    transactions[numeric_columns] = scaler.transform(transactions[numeric_columns])
    transactions.fillna(0, inplace=True)

    # Ordenar y crear secuencias
    transactions = transactions.sort_values(by='timeStamp', ascending=False)
    sequences = transactions.head(sequence_length)

    # Asegurar que las secuencias tengan la longitud correcta
    padded_sequence = np.pad(sequences.to_numpy(), ((0, sequence_length - len(sequences)), (0, 0)), mode='constant', constant_values=0)
    return np.array([padded_sequence])

@tf.function(reduce_retracing=True)
def predict_fn(model, X):
    return model(X, training=False)

def predict_fraudulence(sequences, verbose=2):
    # Cargar el modelo ensamblado exportado previamente
    with open('./exported_models/lstm_ensemble_models.pkl', 'rb') as f:
        models = pickle.load(f)

    # Promediar las predicciones de los modelos en el ensamblaje
    predictions = [predict_fn(model, sequences).numpy() for model in models]

    if verbose > 1:
        print("\n\n\nPredicciones de cada modelo")
        for i in range(len(predictions)):
            pred = (predictions[i] > 0.5).astype("int32")
            print("\nEl modelo ", i+1, " ha predicho una probabilidad de fraude de ", predictions[i][0][0])

    y_pred_avg = np.mean(predictions, axis=0)

    if verbose > 0:
        print("\n\n\nDe media se predice una probabilidad de fraude de ", y_pred_avg[0][0])

    y_pred_avg = (y_pred_avg > 0.5).astype("int32")
    return y_pred_avg

# Ejemplo de uso de las funciones
if __name__ == "__main__":
    if API_KEY is None:
        print("Error, variable de entorno ETHERSCAN_API_KEY no detectada")
        exit(1)
    print("Buscando las transacciones previas de este address...\n\n")

    # Obtener transacciones a partir del hash de una transacción
    address = '0xa8a6497978c264d7ce268d826a834c1d7134d914'  # Reemplaza con el hash de la transacción que deseas consultar
    max_length = 20  # Número máximo de transacciones a devolver

    previous_transactions = get_transactions_by_address(address, max_length=max_length)

    if not previous_transactions.empty:
        print(f"Transacciones previas para el remitente de la transacción {address}:")
        print(previous_transactions)

        with open('./exported_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)




        # Rellenar valores NaN en 'txreceipt_status' con 0 en una sola línea
        previous_transactions['txreceipt_status'].fillna(0, inplace=True)

        # Crear la secuencia de datos
        sequence = create_sequence_dataset(previous_transactions, address, max_length, scaler)


        # Usar el modelo para predecir
        prediction = predict_fraudulence(sequence)

        # Imprimir la predicción
        print(f"Predicción de fraude para la dirección {address}: {'Fraudulenta' if prediction[0][0] == 1 else 'No Fraudulenta'}")
    else:
        print(f"No se encontraron transacciones previas para la dirección {address}.")
