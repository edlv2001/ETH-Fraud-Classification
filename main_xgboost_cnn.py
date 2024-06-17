import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam

# Load the dataset
df = pd.read_csv("./data/Merged_Dataset.csv")


# Separate features and labels
X = df.drop(columns=['Address', 'Flag'])
y = df['Flag']

# Define the DNN model
dnn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the DNN model
dnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Ensure the model has been called by predicting on training data
dnn_model.predict(X_train[:1])

# Extract features from the DNN
layer_name = dnn_model.layers[-3].name  # Get the name of the layer you want to extract features from
intermediate_layer_model = tf.keras.Model(inputs=dnn_model.input,
                                          outputs=dnn_model.get_layer(layer_name).output)

# Call the intermediate model on the training data to extract features
X_train_features = intermediate_layer_model.predict(X_train)
X_test_features = intermediate_layer_model.predict(X_test)

# Apply SMOTE to the features
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)

# Fill missing values if any
X_train_resampled = pd.DataFrame(X_train_resampled).fillna(pd.DataFrame(X_train_resampled).mean())
X_test_features = pd.DataFrame(X_test_features).fillna(pd.DataFrame(X_test_features).mean())

# Initialize the XGBoost model
xgb_model = XGBClassifier(
    max_depth=10,
    n_estimators=300,
    learning_rate=0.2,
    subsample=0.9,
    random_state=42,
    objective='binary:logistic'
)

# Standardize the data
scaler = StandardScaler().fit(X_train_resampled)
X_train_resampled = scaler.transform(X_train_resampled)
X_test_features = scaler.transform(X_test_features)

# Train the XGBoost model
xgb_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = xgb_model.predict(X_test_features)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Show metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")