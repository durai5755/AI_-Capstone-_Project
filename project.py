print("Hello Deva My AI Capstone project started")
import pandas as pd
data= pd.read_csv("Titanic-Dataset.csv")
print(data.head())

import pandas as pd 
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df = df.drop("Cabin", axis=1)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

import matplotlib.pyplot as plt

# Survival Count
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

import matplotlib.pyplot as plt

df.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert Sex to numbers
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Select features
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

