Import pandas as pd

Import numpy as np

Import matplotlib.pyplot as plt

Import seaborn as sns

# Load the dataset

Dataset = pd.read_csv(‘diabetes.csv’)

# Display the first few rows of the dataset

Dataset.head()

# Display the shape of the dataset

Dataset.shape

# Display information about the dataset

Dataset.info()

# Display the descriptive statistics of the dataset

Dataset.describe().T

# Check for missing values in the dataset

Dataset.isnull().sum()

# Visualize the distribution of the target variable

Sns.countplot(x=’Outcome’, data=dataset)

# Visualize the pairwise relationships between variables

Sns.pairplot(data=dataset, hue=’Outcome’)

Plt.show()

# Visualize the correlation matrix

Sns.heatmap(dataset.corr(), annot=True)

Plt.show()

# Create a copy of the dataset

Dataset_new = dataset

# Replace zero values with NaN in selected columns

Dataset_new[[“Glucose”, “BloodPressure”, “SkinThickness”, “Insulin”, “BMI”]] = dataset_new[[“Glucose”, “BloodPressure”, “SkinThickness”, “Insulin”, “BMI”]].replace(0, np.NaN)

# Check for missing values in the updated dataset

Dataset_new.isnull().sum()

# Fill missing values with the mean of each column

Dataset_new[“Glucose”].fillna(dataset_new[“Glucose”].mean(), inplace=True)

Dataset_new[“BloodPressure”].fillna(dataset_new[“BloodPressure”].mean(), inplace=True)

Dataset_new[“SkinThickness”].fillna(dataset_new[“SkinThickness”].mean(), inplace=True)

Dataset_new[“Insulin”].fillna(dataset_new[“Insulin”].mean(), inplace=True)

Dataset_new[“BMI”].fillna(dataset_new[“BMI”].mean(), inplace=True)

# Check for missing values in the updated dataset

Dataset_new.isnull().sum()

# Separate the target variable from the features

Y = dataset_new[‘Outcome’]

X = dataset_new.drop(‘Outcome’, axis=1)

# Split the dataset into training and testing sets

From sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=dataset_new[‘Outcome’])

# Train a logistic regression model

From sklearn.linear_model import LogisticRegression

Model = LogisticRegression()

Model.fit(X_train, Y_train)

# Make predictions on the testing set

Y_predict = model.predict(X_test)

Print(y_predict)

# Evaluate the model using a confusion matrix

From sklearn.metrics import confusion_matrix

Cm = confusion_matrix(Y_test, y_predict)

Print(cm)

# Visualize the confusion matrix

Sns.heatmap(pd.DataFrame(cm), annot=True)

# Calculate the accuracy of the model

From sklearn.metrics import accuracy_score

Accuracy = accuracy_score(Y_test, y_predict)

Accuracy

# Prompt the user to enter input values for prediction

Pregnancies = float(input(“Enter the number of pregnancies: “))

Glucose = float(input(“Enter the glucose level: “))

BloodPressure = float(input(“Enter the blood pressure: “))

SkinThickness = float(input(“Enter the skin thickness: “))

Insulin = float(input(“Enter the insulin level: “))

BMI = float(input(“Enter the BMI: “))

DiabetesPedigreeFunction = float(input(“Enter the diabetes pedigree function: “))

Age = float(input(“Enter the age: “))

# Create an array of input values

Input_values = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age], dtype=float).reshape(1, -1)

# Make a prediction using the trained model

Y_predict = model.predict(input_values)

# Display the input values

Print(“Pregnancies:”, Pregnancies)

Print(“Glucose:”, Glucose)

Print(“BloodPressure:”, BloodPressure)

Print(“SkinThickness:”, SkinThickness)

Print(“Insulin:”, Insulin)

Print(“BMI:”, BMI)

Print(“DiabetesPedigreeFunction:”, DiabetesPedigreeFunction)

Print(“Age:”, Age)

Print()

# Display the prediction

Print(y_predict)

# Display the prediction result

If y_predict == 1:

Print(“Diabetic”)

Else:

Print(“Non-Diabetic”)
