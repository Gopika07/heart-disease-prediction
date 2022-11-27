import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart.csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression(max_iter=1000)

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

import pickle
pickle.dump(model,open("Model.pkl","wb"))