# student-performance-predictor
🎓 Student Performance Prediction using Machine Learning
This project uses machine learning to predict student performance based on various demographic and academic features. The goal is to either:

🎯 Predict average marks (regression)

✅ Classify students as Pass/Fail (classification)

📁 Dataset
Source: Kaggle - Student Performance Dataset
The dataset contains information on:

Gender

Race/Ethnicity

Parental Level of Education

Lunch Type

Test Preparation Course

Scores in Math, Reading, Writing

📌 Objectives
🔍 Explore how factors like gender, study time, and parental education affect performance

🤖 Build models to predict average marks or pass/fail status

📊 Evaluate the performance of Decision Tree and Logistic Regression

🧪 Techniques Used
Task	Algorithm
Classification	Decision Tree, Logistic Regression
Regression	Decision Tree Regressor

🧹 Data Preprocessing
Label Encoding & One-Hot Encoding for categorical data

MinMaxScaler for normalizing score values

Feature engineering:

average score

pass label (1 for average ≥ 40%)

⚙️ Model Building
Split dataset into training and testing sets (80/20)

Trained:

📈 DecisionTreeRegressor for predicting average marks

✅ DecisionTreeClassifier and LogisticRegression for classifying pass/fail

Evaluated using:

Accuracy, Confusion Matrix, Classification Report

MSE, R² Score for regression

📊 Results
Model	Task	Accuracy / R²
Decision Tree	Classification	xx%
Logistic Regression	Classification	xx%
Decision Tree	Regression	R² = xx, MSE = xx

Replace xx with actual values after testing.

📈 Visualization
Confusion matrix using Seaborn heatmap

Decision Tree visualized using plot_tree()

🚀 Future Improvements
Try advanced models: Random Forest, XGBoost

Hyperparameter tuning

Cross-validation

Include UI with Streamlit

🧑‍💻 Author
Aniket Patil
3rd Year BTech CSE (AI), GH Raisoni College, Pune
📌 Interested in Data Science & Generative AI

