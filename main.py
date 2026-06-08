import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression(max_iter=10000)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ROC-AUC Score
probabilities = model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, probabilities)

print("\nROC-AUC Score:", roc)

print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

# Random Forest
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("\n==========================")
print("MODEL COMPARISON")
print("==========================")

print("Logistic Regression :", round(accuracy, 4))
print("Decision Tree       :", round(dt_acc, 4))
print("Random Forest       :", round(rf_acc, 4))
print("\n====================")
print("BEST MODEL")
print("====================")

if accuracy > dt_acc and accuracy > rf_acc:
    print("Logistic Regression")
elif dt_acc > rf_acc:
    print("Decision Tree")
else:
    print("Random Forest")

print("\nProject Executed Successfully")
models = ["Logistic Regression", "Decision Tree", "Random Forest"]
scores = [accuracy, dt_acc, rf_acc]

plt.figure(figsize=(8,5))
plt.bar(models, scores)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("accuracy_comparison.png")
plt.show()


input("Press Enter to exit...")

