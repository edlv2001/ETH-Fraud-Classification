import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

# Cargar el dataset
df = pd.read_csv("../data/Merged_Dataset.csv")

# Separar características y etiqueta
X = df.drop(columns=['Address', 'Flag'])
y = df['Flag']

# Calcular la matriz de correlación y eliminar características altamente correlacionadas
corr_matrix = X.corr().abs()
threshold = 0.9
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
X = X.drop(columns=to_drop)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE para equilibrar las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Rellenar valores faltantes con la media de cada columna en el conjunto de entrenamiento y prueba
X_train_resampled = X_train_resampled.fillna(X_train_resampled.mean())
X_test = X_test.fillna(X_test.mean())

# Normalizar los datos
scaler = PowerTransformer().fit(X_train_resampled)
X_train_resampled = scaler.transform(X_train_resampled)
X_test = scaler.transform(X_test)


def create_model(input_shape, params):
    try:
        model = Sequential()
        model.add(Conv2D(int(params[0]), (1, 1), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Conv2D(int(params[1]), (1, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 1)))  # Cambia el tamaño del pool para evitar reducción negativa

        model.add(Conv2D(int(params[2]), (1, 1), activation='relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(int(params[3]), (1, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 1)))  # Cambia el tamaño del pool para evitar reducción negativa

        model.add(Conv2D(int(params[4]), (1, 1), activation='relu'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=params[5]), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except ValueError as e:
        print(f"Error al crear el modelo: {e}")
        return None


def fitness_function(solution):
    input_shape = (X_train_resampled.shape[1], 1, 1)
    X_train_reshaped = X_train_resampled.reshape(-1, input_shape[0], 1, 1)
    X_test_reshaped = X_test.reshape(-1, input_shape[0], 1, 1)

    model = create_model((input_shape[0], 1, 1), solution)
    if model is None:
        return 0  # Devuelve una puntuación baja si el modelo no se puede construir

    model.fit(X_train_reshaped, y_train_resampled, epochs=5, batch_size=32,
              verbose=0)  # Reduce las épocas para la optimización inicial

    y_pred = (model.predict(X_test_reshaped) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def initialize_population(pop_size, dim):
    population = np.random.rand(pop_size, dim) * 50  # Para inicializar con valores más realistas
    population[:, -1] = np.random.rand(pop_size) * 0.01  # Iniciar learning rate en un rango más pequeño
    return population


def crossover(parent1, parent2):
    child = np.copy(parent1)
    mask = np.random.rand(len(parent1)) > 0.5
    child[mask] = parent2[mask]
    return child


def mutate(solution, mutation_rate):
    for i in range(len(solution)):
        if np.random.rand() < mutation_rate:
            solution[i] = np.random.rand() * 50
    solution[-1] = np.random.rand() * 0.01  # Mantener learning rate en un rango más pequeño
    return solution


def select_parents(population, fitnesses):
    if fitnesses.sum() == 0:
        fitnesses += 1  # Evitar división por cero si todas las fitnesses son 0
    idx1, idx2 = np.random.choice(range(len(population)), size=2, replace=False, p=fitnesses / fitnesses.sum())
    return population[idx1], population[idx2]


def GA_CS_algorithm(pop_size, dim, generations, mutation_rate, pamax, pamin, itermax):
    population = initialize_population(pop_size, dim)
    best_solution = None
    best_fitness = -np.inf

    for generation in range(generations):
        fitnesses = np.array([fitness_function(ind) for ind in population])
        new_population = []

        for i in range(pop_size):
            if np.random.rand() < (pamax - (pamax - pamin) / itermax * generation):
                parent1, parent2 = select_parents(population, fitnesses)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
            else:
                child = np.random.rand(dim) * 50
                child[-1] = np.random.rand() * 0.01  # Mantener learning rate en un rango más pequeño

            new_population.append(child)

        population = np.array(new_population)

        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_solution = population[current_best_idx]

    return best_solution, best_fitness


# Configuración de parámetros para el algoritmo GA-CS
pop_size = 20  # Reduce para pruebas más rápidas
dim = 6  # 5 capas conv + 1 learning rate
generations = 10  # Reduce para pruebas más rápidas
mutation_rate = 0.1
pamax = 0.6
pamin = 0.2
itermax = 50

best_solution = [1.69194590e+01, 1.31151815e+01, 4.57275337e+01, 4.29212607e+01, 2.07839625e+00, 9.22081925e-03]

if best_solution is None:
    # Ejecutar el algoritmo GA-CS para encontrar los mejores hiperparámetros
    print("Comenzando algoritmo Genético")
    best_solution, best_fitness = GA_CS_algorithm(pop_size, dim, generations, mutation_rate, pamax, pamin, itermax)
    print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Utilizar los mejores hiperparámetros encontrados para entrenar el modelo final
input_shape = (X_train_resampled.shape[1], 1, 1)
X_train_reshaped = X_train_resampled.reshape(-1, input_shape[0], 1, 1)
X_test_reshaped = X_test.reshape(-1, input_shape[0], 1, 1)

final_model = create_model((input_shape[0], 1, 1), best_solution)
if final_model is not None:
    checkpoint = ModelCheckpoint('../best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    final_model.fit(X_train_reshaped, y_train_resampled, epochs=50, batch_size=32,
                    validation_data=(X_test_reshaped, y_test), callbacks=[checkpoint])

    # Evaluar el modelo final
    y_pred = (final_model.predict(X_test_reshaped) > 0.5).astype("int32")
    final_accuracy = accuracy_score(y_test, y_pred)
    final_precision = precision_score(y_test, y_pred)
    final_recall = recall_score(y_test, y_pred)
    final_f1 = f1_score(y_test, y_pred)
    final_roc_auc = roc_auc_score(y_test, y_pred)
    final_conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Final model accuracy: {final_accuracy:.4f}")
    print(f"Final model precision: {final_precision:.4f}")
    print(f"Final model recall: {final_recall:.4f}")
    print(f"Final model F1 score: {final_f1:.4f}")
    print(f"Final model ROC AUC: {final_roc_auc:.4f}")
    print("Confusion Matrix:")
    print(final_conf_matrix)
else:
    print("No se pudo construir el modelo final con los mejores hiperparámetros encontrados.")