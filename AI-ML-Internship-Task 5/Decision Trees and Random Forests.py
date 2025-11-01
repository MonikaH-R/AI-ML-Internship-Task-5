import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Configuration ---
DATASET_PATH = 'heart.csv'
OUTPUT_DIR = 'output'
RANDOM_SEED = 42
K_FOLDS = 5

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output folder created at: {OUTPUT_DIR}/")

# --- 0. Data Loading and Preprocessing ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please place it in the project root.")
    exit()

# Handle Categorical Features (One-Hot Encoding)
# 'cp', 'thal', 'ca' are typically categorical/ordinal and benefit from one-hot encoding
# The UCI dataset often has these as numbers, but treating them as categories is common practice.
X = df.drop('target', axis=1)
y = df['target']
X = pd.get_dummies(X, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
print(f"Training set size: {X_train.shape[0]} samples")
print("-" * 50)

# --- 1. Train a Decision Tree Classifier and Visualize the Tree ---
print("Task 1: Training and Visualizing Decision Tree...")
dt_classifier = DecisionTreeClassifier(random_state=RANDOM_SEED)
dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)
initial_accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Initial (Unconstrained) Decision Tree Test Accuracy: {initial_accuracy_dt:.4f}")

# Visualize the tree (limiting depth for a readable image)
plt.figure(figsize=(25, 12))
plot_tree(dt_classifier,
          filled=True,
          rounded=True,
          class_names=['No Disease', 'Disease'],
          feature_names=X.columns.tolist(),
          max_depth=3) # Limit depth for visualization clarity
plt.title("Decision Tree Visualization (Max Depth 3)")
plt.savefig(os.path.join(OUTPUT_DIR, '1_decision_tree_visualization.png'))
plt.close()
print(f"-> Saved: 1_decision_tree_visualization.png in {OUTPUT_DIR}")
print("-" * 50)


# --- 2. Analyze Overfitting and Control Tree Depth ---
print("Task 2: Analyzing Overfitting by Controlling Tree Depth...")
train_scores = []
test_scores = []
depths = range(1, 15)

for max_d in depths:
    dt_tuned = DecisionTreeClassifier(max_depth=max_d, random_state=RANDOM_SEED)
    dt_tuned.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt_tuned.predict(X_train)))
    test_scores.append(accuracy_score(y_test, dt_tuned.predict(X_test)))

# Find optimal depth
optimal_depth_index = np.argmax(test_scores)
optimal_depth = depths[optimal_depth_index]

# Plot training vs. testing accuracy
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Training Accuracy', marker='o')
plt.plot(depths, test_scores, label='Testing Accuracy', marker='o')
plt.axvline(x=optimal_depth, color='r', linestyle='--', label=f'Optimal Depth={optimal_depth}')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs. Max Depth (Overfitting Analysis)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, '2_dt_overfitting_analysis.png'))
plt.close()
print(f"Optimal Max Depth found: {optimal_depth}")
print(f"-> Saved: 2_dt_overfitting_analysis.png in {OUTPUT_DIR}")

# Final Tuned DT
dt_tuned_final = DecisionTreeClassifier(max_depth=optimal_depth, random_state=RANDOM_SEED)
dt_tuned_final.fit(X_train, y_train)
tuned_accuracy_dt = accuracy_score(y_test, dt_tuned_final.predict(X_test))
print(f"Tuned Decision Tree Test Accuracy: {tuned_accuracy_dt:.4f}")
print("-" * 50)


# --- 3. Train a Random Forest and Compare Accuracy ---
print("Task 3: Training Random Forest and Comparing Accuracy...")
# Use a reasonable number of estimators (trees)
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=optimal_depth, random_state=RANDOM_SEED)
rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {accuracy_rf:.4f}")
print("-" * 50)


# --- 4. Interpret Feature Importances ---
print("Task 4: Interpreting Random Forest Feature Importances...")
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sorted_importances.head(10).plot(kind='barh')
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance Score (Gini/Entropy Reduction)")
plt.ylabel("Feature")
plt.gca().invert_yaxis() # Highest importance on top
plt.savefig(os.path.join(OUTPUT_DIR, '4_rf_feature_importances.png'))
plt.close()

print("Top 5 Most Important Features:")
print(sorted_importances.head(5))
print(f"-> Saved: 4_rf_feature_importances.png in {OUTPUT_DIR}")
print("-" * 50)


# --- 5. Evaluate Using Cross-Validation ---
print(f"Task 5: Evaluating Models using {K_FOLDS}-Fold Cross-Validation...")

# Cross-validation for Tuned Decision Tree
dt_cv_scores = cross_val_score(dt_tuned_final, X, y, cv=K_FOLDS, scoring='accuracy')

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=K_FOLDS, scoring='accuracy')

# Create a summary table for output
summary_data = {
    'Model': ['Decision Tree (Tuned)', 'Random Forest'],
    f'Test Accuracy (20% Split)': [tuned_accuracy_dt, accuracy_rf],
    f'Mean CV Accuracy ({K_FOLDS}-fold)': [dt_cv_scores.mean(), rf_cv_scores.mean()],
    'CV Std Dev': [dt_cv_scores.std(), rf_cv_scores.std()]
}
summary_df = pd.DataFrame(summary_data)

print("\n--- Final Model Evaluation Summary ---")
print(summary_df.to_string(index=False, float_format="%.4f"))

# Save the summary table to a CSV file
summary_df.to_csv(os.path.join(OUTPUT_DIR, '5_model_evaluation_summary.csv'), index=False)
print(f"\n-> Saved: 5_model_evaluation_summary.csv in {OUTPUT_DIR}")
print("\nTask complete. All results saved to the 'output' folder.")