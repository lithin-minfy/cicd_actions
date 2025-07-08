# main.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("🌼 Iris Classification Report:\n")
print(report)

# Save report for CML
with open("metrics.txt", "w") as f:
    f.write("### Iris Classification Report\n")
    f.write("```\n" + report + "```")
