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
  dataset.loc[ dataset['Age'] > 16 and dataset['Age'] <= 32, 'Age'] = 1
  dataset.loc[ dataset['Age'] > 32 and dataset['Age'] <= 48, 'Age'] = 2
  dataset.loc[ dataset['Age'] > 48 and dataset['Age'] <= 64, 'Age'] = 3
  dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]