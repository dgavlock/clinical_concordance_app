#!/usr/bin/env python
# coding: utf-8
# Streamlit
import streamlit as st
#from st_aggrid import AgGrid

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

###### APPLICATION START

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.sidebar.write("# Clinical Concordance Options")

st.markdown('''## Microphysiology Systems Database Clinical Concordance Application''')

st.write("## Introduction")
st.write('''It’s in the expressed interest of the MPS-Db to develop advanced tools for the design, recording, exploration, and inference of microphysiology systems (MPS) experiments from users to better aid users in their investigation of complex biological systems. There is a need in the MPS community to evaluate their models’ post-experiment for their ability to recapitulate in vivo or clinical outcomes of drug toxicity. This need is a general one in the clinical translational science space, as all drug intervention research includes a pre-clinical investigation using in vitro model systems.


MPS-Db aims to provide proprietary, advanced inference methods to their users with highly accurate predictions on drug toxicity using the best open-source data clinical toxicity data available. To do so, we have developed test datasets from MPS data if investigate machine learning approaches can be applied to assessing and predicting clinical concordance of MPS models.
''') 

st.write("### The Machine Learning Pipeline")
st.write("We've tested many Machine Learning models on data from the database and determined the best base model estimator for you to start with. That model is a Calibrate Random Forest model.")

with st.expander("Calibrated Random Forest Details"):
    st.write("""
    **Random Forest** models were first developed as an extension of Decision Tree models. Decision tree models break down data examples into classes using feature probabilities. For example, if you were predicting whether or not someone will run outside and had information on the weather, you might first look at how much someone runs when it rains out. Then if all samples haven’t been classified perfectly, you might add another level and ask how much someone runs when it rains and whether (or not) it’s windy. You might find that when you look at both features you can perfectly predict when someone runs outside. But you can see that there are many permutations of decision trees you could initialize (starting with wind, then looking at rain), which is what Random Forests look to solve. Random forests sample different subsets of data and different Decision Tree initializations and aggregates the results to create a final mapping of how the features predict the question you’re asking using majority voting (other methods are also available), to make a robust, highly accurate prediction. Random Forests are less interpretable than their Decision Tree predecessor due to the voting aggregation step but are preferred due to their robust performance on complex datasets.
    """)
    st.write("""**Model calibration** is a method that is widely used in production ML models to make models match was we are seeing in the data, and thus making predictions more accurate and representative of what our prior knowledge tells us. Model calibration is summarized as the following equation:
    
$p ̂=Pr⁡(Y|p ̂)$

Where p ̂ is the predicted probability given by our classifier. This is the simple question, “How many times does our prediction match our previous data for a given event?” So for our model, the question is, “When we predict something is toxic, how many times is it actually toxic?” and if it’s not let’s update the prediction to match what we see in a systematic way. When predictions are binary, we can assess calibration using a reliability curve, which essentially looks at how well predicted probabilities match observed probabilities. When you calibrate you expect your predicted probabilities to move closer towards the perfect calibration line (aka matching observed probabilities). It’s also important to note that after you calibrate you won’t necessarily expect to achieve perfect calibration as you are probably overfitting your model.
""")



file1 = st.file_uploader("Upload Your Dataset Here", accept_multiple_files=False, help="You should upload a dataset that has labels for all observations and any number of features. Labels should be categorized in the following way: Non-Toxic, Toxic, and Higly-Toxic")
#file1 = "../Mimetas MPS dataset for ML analysis.csv"

## Initial Dataset
if file1 != None:
    
    mps_study = st.sidebar.selectbox("Please Select the MPS Study of Interest", ["Clinical Concordance"])
    clinical_metric = st.sidebar.selectbox("Please Select Clinical Data Source of Interest",["LiverTox Database"])
    
    df = pd.read_csv(file1, sep=",", encoding="Latin")

    ## Editible Grid
    st.write(" ### Data Table")
    # grid_return = AgGrid(df, editable=True)
    # new_df = grid_return['data']
    st.write(df)
    
    with st.expander("Data Statistics Table"):
        st.write(df.describe())
        
        mps_join_id = st.sidebar.selectbox("Please Select MPS-Db column to use as an index for Joining the Selected Clinical Data Label with the MPS Study Data", df.columns)

        clin_join_id = st.sidebar.selectbox("Please Select Clinical column to use as an index for Joining the Selected Clinical Data Label with the MPS Study Data", df.columns)

        label_join_method = st.sidebar.radio("Please Select the Method for Joining the Selected Clinical Data Label with the MPS Study Data", ["Left", "Right"])

    ## Initial Features and Prediction Column Selection
    train_features = st.sidebar.multiselect("Select Model Features", list(df.select_dtypes(include=[np.number]).columns))

    avail_features = list(set(df.columns) - set(train_features))

    train_labels = st.sidebar.selectbox("Select Model Prediction Class", avail_features)
    
    split_ratio = st.sidebar.slider("Please select the ratio of data to be held out for testing", 0.1, 0.9, 0.3)
    
    ## NaN Strategy Selection

    nan_choice = st.sidebar.radio("Please Select Your Strategy for Missing Values", ["Drop rows with missing values", "Do nothing"])

    if nan_choice == "Drop rows with missing values":
        df = df.dropna(axis=0)

    train_X = df[train_features].to_numpy()
    X_labels = train_features
    train_y = df[train_labels].to_numpy()
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

    search = GridSearchCV(pipe, param_grid, cv=5).fit(X_train, y_train)

    best_model = search.best_estimator_


    st.metric("Model Accuracy:", '{0:.4g}%'.format(accuracy_score(best_model.predict(X_test), y_test)*100) )

    y_pred = best_model.predict(X_test)
    y_probas = best_model.predict_proba(X_test)
    labels=y_pred

    st.metric("ROC AUC Score:", '{0:.4g}'.format(roc_auc_score(y_test, y_probas, multi_class='ovo')))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, xticklabels=y_labels, yticklabels=y_labels)
    plt.rcParams["figure.figsize"] = (5,2)
    st.write(fig)
    
    ## Sidebar
    st.sidebar.write("#### Save Your Model to Your MPS Db Profile for Future Use")
    st.sidebar.download_button(label="Save Model", data="best_model")
    
    st.write("### Use your Trained Model")
    
    example = X_test
    
    ## Main
    example1 = st.text_input("Insert New Example to predict", str(example))
    st.write("Predicted Value: ", best_model.predict(X_test))
    