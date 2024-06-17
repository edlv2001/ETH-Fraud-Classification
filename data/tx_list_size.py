import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('Merged_Dataset.csv')

# Calcular la media
media = df['total_transactions_including_tnx_to_create_contract'].mean()

# Calcular la mediana
mediana = df['total_transactions_including_tnx_to_create_contract'].median()
desviacion_media = (df['total_transactions_including_tnx_to_create_contract'] - df['total_transactions_including_tnx_to_create_contract'].mean()).abs().mean()

# Imprimir los resultados
print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Desviación Media: {desviacion_media}")
# Definir el umbral
umbral = 40  # Reemplaza este valor con el umbral que desees

# Contar cuántos valores superan el umbral
valores_superiores_al_umbral = df[df['total_transactions_including_tnx_to_create_contract'] > umbral].shape[0]

print(f"Número de valores que superan el umbral {umbral}: {valores_superiores_al_umbral}")
