import pandas as pd
import matplotlib.pyplot as plt

# Lee el archivo CSV
df = pd.read_csv('Merged_Dataset.csv')

# Reemplaza los valores de la columna 'Flag' para una mejor visualización
df['Flag'] = df['Flag'].map({0: 'No Fraude', 1: 'Fraude'})

# Cuenta el número de elementos en cada clase de la columna 'Flag'
conteo_clases = df['Flag'].value_counts()

# Crea el gráfico de barras
plt.figure(figsize=(10, 6))
conteo_clases.plot(kind='bar')
plt.title('Número de direcciones por clase')
plt.xlabel('Clase')
plt.ylabel('Número de direcciones')
plt.xticks(rotation=45)
plt.show()
