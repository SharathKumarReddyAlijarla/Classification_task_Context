"""
Fake Instagram Profile Detection - Updated Baseline Model.

Author: [Sharath Alijarla]
Date: March 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import random
import optuna
from optuna.samplers import TPESampler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================
# Set SEED for reproducibility
# ==============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
sampler = TPESampler(seed=SEED)

# ========================================
# Load datasets
# ========================================
instagram_df_train = pd.read_csv('train.csv')  # Load training dataset
instagram_df_test = pd.read_csv('test.csv')  # Load testing dataset

# Check for missing values in training dataset
print(instagram_df_train.isnull().sum())

# Visualize class distribution in training data
sns.countplot(x=instagram_df_train['fake'])
plt.title("Class Distribution in Training Data")
plt.show()

# ========================================
# Prepare training and testing data
# ========================================
X_train = instagram_df_train.drop(columns=['fake'])  # Extract features from training data
X_test = instagram_df_test.drop(columns=['fake'])  # Extract features from testing data
y_train = instagram_df_train['fake']  # Extract target variable from training data
y_test = instagram_df_test['fake']  # Extract target variable from testing data

# Standardize features to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode target variable for multi-class classification
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# =========================================
# OPTUNA HYPERPARAMETER TUNING
# =========================================
def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    Trains a neural network with different hyperparameters and evaluates validation accuracy.
    """
    model = Sequential()
    
    # Define input layer with dynamic number of units
    n_units_in = trial.suggest_int('units_input', 32, 256, step=32)
    model.add(Dense(n_units_in, activation='relu', input_dim=11))
    model.add(BatchNormalization())  # Normalize activations
    
    # Define hidden layers dynamically based on trial suggestions
    n_layers = trial.suggest_int('n_layers', 1, 4)
    for i in range(n_layers):
        n_units = trial.suggest_int(f'n_units_l{i}', 32, 256, step=32)
        dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5, step=0.1)
        model.add(Dense(n_units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Define output layer with softmax activation for classification
    model.add(Dense(2, activation='softmax'))
    
    # Define optimizer and learning rate dynamically
    learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
    optimizer = AdamW(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks for early stopping and learning rate adjustment
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    
    # Train model silently to find best parameters
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=50,
                        batch_size=8,
                        verbose=0,
                        callbacks=[early_stop, lr_scheduler])
    
    # Return best validation accuracy to guide optimization
    val_acc = max(history.history['val_accuracy'])
    return val_acc

# Run Optuna optimization to find best hyperparameters
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=20)

# Display best hyperparameters found
print('Best Hyperparameters:')
print(study.best_params)

# =========================================
# TRAIN BEST MODEL USING OPTIMIZED HYPERPARAMETERS
# =========================================
best_params = study.best_params

# Build best model using optimal parameters
model = Sequential()
model.add(Dense(best_params['units_input'], activation='relu', input_dim=11))
model.add(BatchNormalization())

for i in range(best_params['n_layers']):
    model.add(Dense(best_params[f'n_units_l{i}'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(best_params[f'dropout_l{i}']))

model.add(Dense(2, activation='softmax'))

optimizer = AdamW(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train best model
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=50,
                    batch_size=8,
                    callbacks=[early_stop, lr_scheduler],
                    verbose=1)

# =========================================
# EVALUATION & VISUALIZATION
# =========================================

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Generate predictions
predicted = model.predict(X_test)
predicted_value = np.argmax(predicted, axis=1)
test_value = np.argmax(y_test, axis=1)

# Print classification report
print("Classification Report:\n")
print(classification_report(test_value, predicted_value))

# Display confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(test_value, predicted_value), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
