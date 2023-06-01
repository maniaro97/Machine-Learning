# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:41:05 2023

@author: HP
"""

import streamlit as st
import pandas as pd
import os

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

import pycaret
from pycaret.classification import setup, compare_models, pull, save_model, ClassificationExperiment
from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment

st.title("Machine Learning Application using Classification and Regression Models")

if os.path.exists("sourcev.csv"):
    df = pd.read_csv("sourcev.csv",index_col=None)

with st.sidebar:
    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*cG6U1qstYDijh9bPL42e-Q.jpeg")
    st.header("Welcome to the Application!")
    st.subheader("This is made for learning machine models. You can do both classification and regression analysis here.")
    st.caption("Choose your parameters below to work on the application.")
    choose=st.radio(":coffee:",["Dataset","Analysis","Training","Download"])
    st.info("I have made this application which helps in building automated machine learning models using streamlit, pandas, pandas_profiling(for EDA) and pycaret library. Hope ypu like it! :)")
    
if choose=="Dataset":
    st.write("Please upload your dataset here.")
    dataset_value = st.file_uploader("Upload here")
    
    if dataset_value:
        df = pd.read_csv(dataset_value, index_col=None)
        df.to_csv("sourcev.csv", index = None)
        st.dataframe(df)

if choose=="Analysis":
    st.subheader("Perform profiling on Dataset")
    if st.sidebar.button("Do Analysis"):
        profile_report = df.profile_report() 
        st_profile_report(profile_report)
    
if choose=="Training":
    st.header("Start Training your Model now.")
    choice = st.sidebar.selectbox("Select your Technique:", ["Classification","Regression"])
    target = st.selectbox("Select you Target Variable",df.columns)
    if choice=="Classification":
        if st.sidebar.button("Classification Train"):
            s1 = ClassificationExperiment()
            s1.setup(data=df, target=target)
            setup_df = s1.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
            
            best_model1 = s1.compare_models()
            compare_model = s1.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
            
            best_model1
            s1.save_model(best_model1,"Machine Learning Model")
    else:
        if st.sidebar.button("Regression Train"):
            s2 = RegressionExperiment()
            s2.setup(data=df, target=target)
            setup_df = s2.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
            
            best_model2 = s2.compare_models()
            compare_model = s2.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
            
            best_model2
            s2.save_model(best_model2,"Machine Learning Model")

if choose =="Download":
    with open("Machine Learning model.pkl",'rb') as f:
        st.caption("Download your model from here:")
        st.download_button("Download the file",f,"Machine Learning model.pkl")
        
    
    
    
    
    





