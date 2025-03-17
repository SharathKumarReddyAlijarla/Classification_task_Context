"""
Fake Instagram Profile Detection - Custom Model
This script implements data preprocessing, feature selection, and multiple models 
(Logistic Regression, Random Forest, and Neural Network) to classify fake Instagram profiles.

Author: [Sharath Alijarla]
Date: March 2025
"""

# ==============================
# Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random

# ==============================
# Set SEED for Reproducibility
# ==============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ==============================
# Load Datasets
# ==============================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Display missing values count in training data
print("Missing values in train dataset:\n", train_df.isnull().sum())

# ==============================
# Visualizing Class Distribution
# ==============================
sns.countplot(x=train_df["fake"])
plt.title("Class Distribution in Training Data")
plt.show()

# ==============================
# Handling Class Imbalance
# Upsampling the minority class to match majority class count
# ==============================
majority_class = train_df[train_df["fake"] == 0]
minority_class = train_df[train_df["fake"] == 1]

# Upsampling the minority class
minority_upsampled = resample(
    minority_class, replace=True, n_samples=len(majority_class), random_state=SEED
)

# Combine the balanced dataset
train_df = pd.concat([majority_class, minority_upsampled])

# Shuffle data after balancing
train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ==============================
# Feature Selection - Removing Low Variance Features
# ==============================
low_variance_threshold = 0.01  # Features with variance below this will be removed
low_variance_features = train_df.drop(columns=["fake"]).std() < low_variance_threshold

# Dropping low variance features
train_df = train_df.drop(columns=low_variance_features[low_variance_features].index)
test_df = test_df.drop(columns=low_variance_features[low_variance_features].index)

# ==============================
# Feature Selection - Removing Highly Correlated Features
# ==============================
correlation_matrix = train_df.corr()
high_corr_features = set()
corr_threshold = 0.9  # Features with correlation > 0.9 will be removed

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > corr_threshold:
            colname = correlation_matrix.columns[i]
            high_corr_features.add(colname)

# Dropping highly correlated features
train_df = train_df.drop(columns=high_corr_features)
test_df = test_df.drop(columns=high_corr_features)

# ==============================
# Feature Selection - Using Random Forest Importance
# ==============================
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(train_df.drop(columns=["fake"]), train_df["fake"])

# Selecting features with importance greater than 0.01
feature_importances = pd.Series(rf.feature_importances_, index=train_df.drop(columns=["fake"]).columns)
important_features = feature_importances[feature_importances > 0.01].index

# Keep only important features
train_df = train_df[important_features.tolist() + ["fake"]]
test_df = test_df[important_features.tolist() + ["fake"]]

# ==============================
# Splitting Features & Labels
# ==============================
X_train = train_df.drop(columns=["fake"])
y_train = train_df["fake"]
X_test = test_df.drop(columns=["fake"])
y_test = test_df["fake"]

# ==============================
# Standardize Features
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Logistic Regression Model
# ==============================
log_reg = LogisticRegression(random_state=SEED)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Print Classification Report
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_log))

# ==============================
# Random Forest Model
# ==============================
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Print Classification Report
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# ==============================
# Callbacks for Neural Network
# ==============================
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

# ==============================
# Neural Network Model
# ==============================
model = Sequential(
    [
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile Model
model.compile(optimizer=AdamW(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=8, verbose=1, callbacks=[lr_scheduler, early_stopping])

# ==============================
# Evaluate Model
# ==============================
loss, accuracy = model.evaluate(X_test, y_test)
y_pred_nn = (model.predict(X_test) > 0.5).astype(int)

print(f"Neural Network Accuracy: {accuracy:.4f}")
print("Neural Network Report:\n", classification_report(y_test, y_pred_nn))

# ==============================
# Confusion Matrix
# ==============================
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# Plot Loss and Accuracy Curves
# ==============================
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
