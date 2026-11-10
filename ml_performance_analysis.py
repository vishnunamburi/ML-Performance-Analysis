
# ============================================================
# ML LAB MINI PROJECT
# Performance Analysis of Machine Learning Models
# Models:
# 1. Linear Regression
# 2. Decision Tree Regression
# 3. Random Forest Regression
# Dataset: California Housing Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("\nLOADING CALIFORNIA HOUSING DATASET...")

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------

print("\nCHECKING MISSING VALUES:")

print(df.isnull().sum())

# Remove duplicate records if any
df = df.drop_duplicates()

# Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget Variable: Median House Value")

# ------------------------------------------------------------
# 3. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING AND TESTING DATA:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------
# 4. INITIALIZE MACHINE LEARNING MODELS
# ------------------------------------------------------------

models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
}

# ------------------------------------------------------------
# 5. TRAIN MODELS AND MAKE PREDICTIONS
# ------------------------------------------------------------

results = []
predictions = {}

print("\n========== MODEL TRAINING STARTED ==========")

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"{name} Training Completed!")

# ------------------------------------------------------------
# 6. PERFORMANCE COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL PERFORMANCE ==========")

print(results_df.to_string(index=False))

# Save performance results
results_df.to_csv("model_performance_results.csv", index=False)

# Identify best model
best_model = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\n========== BEST MODEL ==========")

print("Best Model:", best_model["Model"])
print("R2 Score:", round(best_model["R2 Score"], 4))
print("RMSE:", round(best_model["RMSE"], 4))

# ------------------------------------------------------------
# 7. VISUALIZATION 1: MODEL PERFORMANCE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("R2 Score Comparison of ML Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("r2_score_comparison.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 8. VISUALIZATION 2: ERROR COMPARISON
# ------------------------------------------------------------

metrics = ["MAE", "MSE", "RMSE"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, metric in enumerate(metrics):

    sns.barplot(
        data=results_df,
        x="Model",
        y=metric,
        ax=axes[i]
    )

    axes[i].set_title(f"{metric} Comparison")
    axes[i].tick_params(axis="x", rotation=20)

plt.tight_layout()

plt.savefig("error_comparison.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 9. VISUALIZATION 3: ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, y_pred) in enumerate(predictions.items()):

    axes[i].scatter(
        y_test,
        y_pred,
        alpha=0.3
    )

    # Reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    axes[i].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--"
    )

    axes[i].set_title(name)

    axes[i].set_xlabel("Actual Values")
    axes[i].set_ylabel("Predicted Values")

plt.suptitle("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 10. VISUALIZATION 4: RESIDUAL PLOTS
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, y_pred) in enumerate(predictions.items()):

    residuals = y_test.values - y_pred

    axes[i].scatter(
        y_pred,
        residuals,
        alpha=0.3
    )

    axes[i].axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted Values")
    axes[i].set_ylabel("Residual Errors")

plt.suptitle("Residual Error Analysis")

plt.tight_layout()

plt.savefig("residual_analysis.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 11. FEATURE IMPORTANCE
# ------------------------------------------------------------

# Decision Tree Feature Importance
tree_model = models["Decision Tree"]

tree_importance = pd.Series(
    tree_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# Random Forest Feature Importance
forest_model = models["Random Forest"]

forest_importance = pd.Series(
    forest_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

tree_importance.plot(
    kind="bar",
    ax=axes[0]
)

axes[0].set_title("Decision Tree Feature Importance")
axes[0].set_ylabel("Importance")

forest_importance.plot(
    kind="bar",
    ax=axes[1]
)

axes[1].set_title("Random Forest Feature Importance")
axes[1].set_ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 12. SAVE PREDICTION RESULTS
# ------------------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Linear Regression": predictions["Linear Regression"],
    "Decision Tree": predictions["Decision Tree"],
    "Random Forest": predictions["Random Forest"]
})

prediction_df.to_csv(
    "model_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 13. FINAL PROJECT SUMMARY
# ------------------------------------------------------------

print("\n============================================")
print("       FINAL PROJECT SUMMARY")
print("============================================")

print("\nDataset: California Housing Dataset")
print("Total Records:", len(df))
print("Number of Features:", X.shape[1])

print("\nModels Evaluated:")

for name in models:
    print("-", name)

print("\nPerformance Results:")
print(results_df.round(4).to_string(index=False))

print("\nBest Performing Model:", best_model["Model"])

print("\nAll graphs and CSV files saved successfully!")

print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("============================================")

