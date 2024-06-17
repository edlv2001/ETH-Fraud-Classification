import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lee el archivo CSV
df = pd.read_csv('Merged_Dataset.csv')
# Elimina la columna 'Address'
df = df.drop(columns=['Address'])

# Calcula la matriz de correlación
matriz_correlacion = df.corr()

# Muestra la matriz de correlación con nombres completos
plt.figure(figsize=(14, 12))
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación de los nombres de las variables
plt.yticks(rotation=0)  # Asegura que las etiquetas en el eje y estén correctamente alineadas
plt.title('Matriz de Correlación de las Columnas')
plt.tight_layout()  # Ajusta el diseño para evitar que las etiquetas se corten
plt.show()

# Imprimir la matriz de correlación
print(matriz_correlacion)
