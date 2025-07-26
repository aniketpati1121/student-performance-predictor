import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Example data
data = pd.DataFrame({
    'study_time': [2, 3, 1, 4],
    'parent_education': [1, 2, 1, 3],
    'gender': [1, 0, 1, 0],
    'passed': [0, 1, 0, 1]
})

X = data.drop("passed", axis=1)
y = data["passed"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save correctly
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
