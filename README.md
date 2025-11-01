# AI-ML-Internship-Task-5
Heart Disease Classification: Decision Trees and Random Forests

**Objective:**
This project implements and compares two fundamental machine learning algorithms—Decision Trees and Random Forests—for a classification task, specifically predicting the presence of heart disease based on clinical features.

The analysis focuses on understanding how tree-based models work, diagnosing and mitigating overfitting, and
evaluating model performance using cross-validation.

 **Features and Analysis Steps**

1. **Decision Tree Training & Visualization:** Trains an initial Decision Tree classifier and saves a visual representation of its structure.

2. **Overfitting Analysis: **Analyzes the trade-off between model complexity and performance by tuning the tree's max_depth parameter.

3. **Random Forest Comparison: **Trains an ensemble Random Forest Classifier to improve predictive accuracy and stability.

4. **Feature Importance Interpretation: **Extracts and visualizes the most impactful features identified by the Random Forest model.

5. **Cross-Validation: **Evaluates the final models robustly using K-Fold Cross-Validation.

** Prerequisites**
To run this script, you need Python and the following libraries:

1. Python 3.x
2. pandas
3. numpy
4. scikit-learn
5. matplotlib


**You can install all necessary libraries using pip:**
pip install pandas numpy scikit-learn matplotlib


** Project Setup and Execution**

**1. Data**

The script requires a dataset named heart.csv (typically the UCI Heart Disease dataset) placed in the root directory of the project.

**2. File Structure**

Ensure your project directory looks like this before running:

Heart_Disease_Analysis/
├── heart.csv
└── decision_tree_analysis.py  


**3. Run the Script**

**Execute the Python file from your terminal:**

python decision_tree_analysis.py

The script will automatically create the output/ folder and populate it with the results.

 
**File Name and Description**

**1_decision_tree_visualization.png
** A graphic visualization of the Decision Tree (limited depth for readability).

**2_dt_overfitting_analysis.png**
A line plot showing how training and testing accuracy change with max_depth, identifying the optimal complexity.

**4_rf_feature_importances.png**
A bar chart displaying the top 10 most important features for prediction according to the Random Forest model.

**5_model_evaluation_summary.csv**
A CSV file summarizing the final performance metrics (Test Accuracy and Mean Cross-Validation Accuracy) for both the Tuned Decision Tree and Random Forest.

 **Code Details
**

The analysis is performed within the single file: decision_tree_analysis.py. Key sections include:

1. **Data Preprocessing: ** Handles loading the data and performing One-Hot Encoding on relevant categorical features (sex, cp, thal, etc.).

2. **Model Instantiation:** Uses DecisionTreeClassifier and RandomForestClassifier from sklearn.tree and sklearn.ensemble.

3. **Evaluation: ** Utilizes accuracy_score and cross_val_score for metric calculation.
