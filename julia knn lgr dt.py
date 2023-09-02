from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the Decision Tree Classifier
start_time = time.time()
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_execution_time = time.time() - start_time

print("Decision Tree Classifier:")
print(f"Accuracy: {dt_accuracy:.2f}")
print(f"Execution Time: {dt_execution_time:.4f} seconds\n")

# Evaluate Logistic Regression
start_time = time.time()
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_execution_time = time.time() - start_time

print("Logistic Regression:")
print(f"Accuracy: {lr_accuracy:.2f}")
print(f"Execution Time: {lr_execution_time:.4f} seconds\n")

# Evaluate K-Nearest Neighbors (KNN) Classifier
start_time = time.time()
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_execution_time = time.time() - start_time

print("K-Nearest Neighbors (KNN) Classifier:")
print(f"Accuracy: {knn_accuracy:.2f}")
print(f"Execution Time: {knn_execution_time:.4f} seconds\n")
