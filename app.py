import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotData import plotFigures
from makeRandomData import randomData
from logisticRegression import LogisticRegression

st.set_page_config(
   page_title="ML Playground",
   page_icon="🤖",
   layout="wide"
)
st.title('ML-Playground')
st.write('A website used to display algorithms that have been developed from scratch for the purpose of classification of various tasks')
st.text('Note: Play around with the parameters in the sidebar to see how an algorithm classifies data and how it adapts to/ performs on the sample data displayed.')

#placeholders for data needed
if "data" not in st.session_state:
    st.session_state["data"] = (np.zeros((5,2)), np.zeros((5)))
if "test_size" not in st.session_state:
    st.session_state["test_size"] = 10
if "num_iter" not in st.session_state:
    st.session_state["num_iter"] = 100
if "lr" not in st.session_state:
    st.session_state["lr"] = 0.01
if "trainingdata" not in st.session_state:
    st.session_state["trainingdata"] = (np.zeros((5,2)), np.zeros((5)))    
if "testingdata" not in st.session_state:
    st.session_state["testingdata"] = (np.zeros((5,2)), np.zeros((5)))
if "scatterPlot" not in st.session_state:
    st.session_state["scatterPlot"] = 0

with st.sidebar:
    #dataset configurations
    header = st.header("Dataset")
    data_button = st.button("Create Random Dataset")
    
    #algorithm configurations
    header = st.header("Algorithm")
    algorithm = st.selectbox(
        'Algorithm to use',
        ('Logistic Regression', ))    
    
    #ML model configurations
    header = st.header("Model HyperParameters")
    test_size = st.slider("Percentage of data to be kept for testing", min_value=10, max_value=99)
    num_iter = st.number_input("Number of iterations to run the model through for training", min_value=10)
    lr = st.number_input("Learning Rate", min_value=0.01, max_value=10.0)

    submit_button = st.button("Predict")

col1, col2 = st.columns(2)

    
#if data_button is true
#that means a request for a random 
#generation of data has been sent
if data_button:
    
    data=randomData()
    X,y=data.createData()
    st.session_state["data"]=(X,y)
    
    X_train, X_test, y_train, y_test = data.split_data(X, y, test_size=test_size*0.01)
    
    st.session_state["trainingdata"] = (X_train, y_train)
    st.session_state["testingdata"] = (X_test, y_test)
    
plots = plotFigures()
inputPlot = plots.scatterPlot(st.session_state["data"][0],st.session_state["data"][1])
st.session_state["scatterPlot"]=inputPlot
    
if submit_button:
    with st.spinner():
        with col1:
            model = LogisticRegression()
            X_train, y_train = st.session_state["trainingdata"]
            X_test, y_test = st.session_state["testingdata"]

            cost, theta = model.train(X_train, y_train, X_test, y_test, num_iter=num_iter, learning_rate=lr)
            st.subheader("Cost and Accuracy vs Iterations")
            st.line_chart(pd.DataFrame({"Cost": cost["cost"], 
                                "Test set accuracy" :cost["predictions"],
                                "Training set accuracy" :cost["training_accuracy"]
                               }))

            boundary = plots.classificationBoundary(st.session_state["data"][0], st.session_state["data"][1], theta)
            st.session_state["scatterPlot"]=boundary

            st.subheader("Metrics")
            st.metric("Accuracy on test(unseen) set", value = f'{round(np.mean(cost["predictions"])*100,2)} %')
            st.metric("Accuracy on training set", value = f'{round(np.mean(cost["training_accuracy"])*100,2)} %')
        
with col2:
    st.subheader("Dataset")
    plot=st.pyplot(st.session_state["scatterPlot"])