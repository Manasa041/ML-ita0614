import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("cars.csv")  
print("First five rows:")
print(data.head())
print("Basic statistics:")
print(data.describe())
print("Columns and their data types:")
print(data.dtypes)

null_columns = data.columns[data.isnull().any()]
for column in null_columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

correlation_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="hot")
plt.title("Correlation Heatmap")
plt.show()

X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
