# 📸 Fake Instagram Profile Detection

## 🚀 Task Description

This project aims to build robust models to detect fake Instagram profiles using two distinct approaches:

1. **Updated Baseline Model**: Enhanced Deep Neural Network with advanced hyperparameter tuning and regularization techniques.
2. **Custom Model**: Combines Traditional Machine Learning models (Logistic Regression, Random Forest) with an Enhanced Neural Network architecture.

---

## 📊 Model Overview

### 1️⃣ Updated Baseline Model (Advanced DNN)

**Techniques Used:**
- **Optuna Hyperparameter Tuning**: Optimizes number of layers, units, dropout rate, learning rate.
- **Batch Normalization & Dropout**: Stabilizes learning, faster convergence, improved generalization.
- **EarlyStopping & ReduceLROnPlateau**: Prevents overfitting, dynamically adjusts learning rate.
- **AdamW Optimizer**: Decouples weight decay for better regularization.
- **Proper Seeding**: Ensures reproducibility of results.

**Impact:**
- Improved accuracy, generalization, and stability.
- Reduced overfitting and better model convergence.

---

### 2️⃣ Custom Model (Traditional ML + Enhanced Neural Network)

**Preprocessing:**
- Checked for missing values.
- Handled class imbalance using **upsampling** of minority (fake) class.

**Feature Selection Techniques:**
1. **Low Variance Removal**: Eliminated features with near-zero variance.
2. **High Correlation Removal**: Removed features with correlation > 0.9.
3. **Random Forest Feature Importance**: Selected significant features (importance > 0.01).

**Models Implemented:**
- **Logistic Regression**: Provides baseline performance.
- **Random Forest Classifier**: Handles feature interactions & non-linearity well.
- **Enhanced Neural Network**:
  - 4 Dense Layers: 256 → 64 → 64 → 32 → Output.
  - Batch Normalization after each Dense layer.
  - Dropout (0.3) to prevent overfitting.
  - AdamW Optimizer with EarlyStopping & ReduceLROnPlateau.

**Evaluation:**
- Precision, Recall, F1-score.
- Confusion Matrix.
- Loss & Accuracy Curves.

---

## 🛠️ How to Run the Code

### 1️⃣ Install Dependencies:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Prepare Data:

Place the following files in your project directory:

- `train.csv`
- `test.csv`

---

### 3️⃣ Run Models:

```bash
# For Updated Baseline Model:
python updated_baseline_model.py

# For Custom Model:
python custom_model_pipeline.py
```
## 📝 Coding Style Guidelines

| Feature             | Details                                                      |
|---------------------|--------------------------------------------------------------|
| **PEP8 Compliant**   | Proper indentation, spacing, and clear naming conventions    |
| **Modular Code**     | Functions for preprocessing, training, evaluation            |
| **Reproducibility**  | Seeded `numpy`, `random`, `tensorflow` for consistent results |
| **Clear Documentation** | Comments & docstrings for each block and function         |
| **Visualization**    | Confusion Matrix, Training Curves with Matplotlib/Seaborn    |

## 📈 Results Summary

| Model                    | Precision                  | Recall                  | F1-Score                 
|-------------------------|----------------------------|------------------------|-------------------------|
| **Logistic Regression**  | 90  | Moderate recall         | Moderate                 |
| **Random Forest**        | 90    | Better recall & precision| Improved                | 
| **Enhanced Neural Network** | 92    | 92     | 92     |
| **Updated Baseline (Optuna)** | 93  | 93    | 92 | 

## 📄 Files Included

| File Name                 | Description                                           |
|--------------------------|-------------------------------------------------------|
| `custom_model_pipeline.py`| Custom Model pipeline (ML + Enhanced Neural Network)  |
| `updated_baseline_model.py`| Enhanced Baseline Deep Neural Network Model         |
| `train.csv`, `test.csv`   | Input datasets                                        |
| `requirements.txt`       | Required dependencies                                 |
| `README.md`               | Project overview & instructions                      |

