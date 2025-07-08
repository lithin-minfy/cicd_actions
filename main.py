# main.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and report
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)

# Save for CML comment
with open("metrics.txt", "w") as f:
    f.write("### Iris Classification Report\n")
    f.write("```\n")
    f.write(report)
    f.write("```")
