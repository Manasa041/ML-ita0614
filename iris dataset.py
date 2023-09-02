import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# a) Read the IRIS.csv dataset using Pandas
iris_data = pd.read_csv("IRIS.csv")  # Replace "IRIS.csv" with your dataset file path

# b) Plot sepal_width versus sepal_length and color species
plt.figure(figsize=(10, 6))
colors = {'setosa': 'r', 'versicolor': 'g', 'virginica': 'b'}
for species, color in colors.items():
    species_data = iris_data[iris_data['species'] == species]
    plt.scatter(species_data['sepal_length'], species_data['sepal_width'], c=color, label=species)
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.title('Sepal Width vs Sepal Length')
plt.show()

# c) Split the data into training and testing sets
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# d) Fit the data to a machine learning model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# e) Predict the model with new test data [5, 3, 1, 0.3]
new_data = np.array([[5, 3, 1, 0.3]])
prediction = model.predict(new_data)
print("Predicted Species:", prediction[0])
