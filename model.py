import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
combine = [train_df, test_df]

# Drop features with fewer data points

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Creating new title feature
for dataset in combine:
  dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
  dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
    'Don', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona' ], 'Rare')
  dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
  dataset['Title'] = dataset['Title'].map(title_mapping)
  dataset['Title'] = dataset['Title'].fillna(0)

# Now we can safely drop Name and PassengerID features
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# Converting a categorical feature
for dataset in combine:
   dataset['Sex'] = dataset['Sex'].map( { 'female': 1, 'male': 0 }).astype(int)

# Completing missing age values
guess_ages = np.zeros((2,3))

for dataset in combine:
  for i in range(0, 2):
    for j in range(0, 3):
      guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
      age_guess = guess_df.median()
      guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5
    
  for i in range(0,2):
    for j in range(0, 3):
      dataset.loc[ (dataset.Age.isnull() & (dataset.Sex == i) & (dataset.Pclass == j+1)), 'Age'] = guess_ages[i,j]
  
  dataset['Age'] = dataset['Age'].astype(int)

# Create age bands and determine correlations with survived
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

for dataset in combine:
  dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
  dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
  dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
  dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
  dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
  dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
  dataset['IsAlone'] = 0
  dataset.loc[ dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
  dataset.Embarked = dataset['Embarked'].fillna(freq_port)

for dataset in combine:
  dataset['Embarked'] = dataset['Embarked'].map( { 'S': 0, 'C': 1, 'Q': 2 } ).astype(int)



for dataset in combine:
  dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
  dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
  dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
  dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
  dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
  dataset['Fare'] = dataset['Fare'].astype(int)
  dataset['Title'] = dataset['Title'].astype(int)

combine = [train_df, test_df]

# Model, predict and solve

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Logistic regression to explore correlation between features and label

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

# Support Vector Machine

svc = SVC(kernel='rbf',C=10000.0, random_state=42)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# KNeighbors

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Decision Tree

decision_tree = DecisionTreeClassifier(min_samples_split=2, random_state=42)
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# AdaBoost

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, Y_train)
Y_pred = ada.predict(X_test)
acc_ada = round(ada.score(X_train, Y_train) * 100, 2)


# Model Evaluation
models = pd.DataFrame({
  'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'RandomForest', 'Naive Bayes', 'Perceptron', 'Decision Tree', 'AdaBoost'],
  'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_decision_tree, acc_ada]
})

print models.sort_values(by='Score', ascending=False)