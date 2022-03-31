#!/usr/bin/env python
# coding: utf-8
# Streamlit
import streamlit as st

# Standard Data Science Importants for Plotting and Transformations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn Imports
## Utilities & Warning
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

## Proprocessing & Pipelining
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

## Model Selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

## Model Calibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

## Feature Selection
from sklearn.feature_selection import SelectKBest

## Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

## Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

st.write('''## Hello, welcome to the Microphysiology Systems Database Clinical Concordance application!''')

st.write("## Introduction")
st.write('''In this application you can add data and define a Machine Learning model that will predict drug toxicity. You can use this model in the future to predict the toxicity if a drug has induced toxicity in your microphysiology system!''') 

st.write("### The Machine Learning Pipeline")
st.write("We've tested many Machine Learning models on data from the database and determined the best base model estimator for you to start with. That model is a Calibrate Random Forest model.")

#st.file_uploader("Upload Your Dataset Here", accept_multiple_files=True, help="You should upload a dataset that has labels for all observations and any number of features. Labels should be categorized in the following way: Non-Toxic, Toxic, and Higly-Toxic")

## Initial Dataset
df = pd.read_csv("../Mimetas MPS dataset for ML analysis.csv", sep=",", encoding="Latin")
st.dataframe(df.columns)

## Make Train-Test Split
train = df[df.Set == "Training"]
validation_test = df[df.Set == "Test"]

split_ratio = st.slider("Please select a training split ratio", 0.1, 0.9, 0.3)


## Initial Features and Prediction Column Selection

train_features = st.multiselect("Select Model Features", train.select_dtypes(include=[np.number]).columns)

avail_features = list(set(df.columns) - set(train_features))

train_labels = st.selectbox("Select Model Prediction Class", avail_features)

train_X = train[train_features].to_numpy()
X_labels = train_features
train_y = train[train_labels].to_numpy()
y_labels = ["Highly Toxic", "Toxic", "Non-Toxic"]

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=split_ratio, random_state=42)

## Setting up Model & Params
calibrated_forest = CalibratedClassifierCV(
    base_estimator=RandomForestClassifier(n_estimators=20, random_state=42), ensemble=True)

### Params
param_grid = {
    'select__k': [1,2],
    'model__base_estimator__max_depth': [2, 4, 6, 8]
}

## Making Pipeline
pipe = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('select', SelectKBest()),
    ('model', calibrated_forest)])

search = GridSearchCV(pipe, param_grid, cv=5).fit(train_X, train_y)

best_model = search.best_estimator_


st.metric("Model Accuracy:", '{0:.4g}%'.format(accuracy_score(best_model.predict(X_test), y_test)*100) )

y_pred = best_model.predict(X_test)
y_probas = best_model.predict_proba(X_test)
labels=y_pred

roc_auc_score(y_test, y_probas, multi_class='ovo')

roc_auc_score(train_y, best_model.predict_proba(train_X), average='macro', multi_class='ovo')

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, xticklabels=y_labels, yticklabels=y_labels)
st.write(fig)