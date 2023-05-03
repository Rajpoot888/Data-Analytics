#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier as xgb
from sklearn.ensemble import RandomForestClassifier as rfc


# print(st.__dict__)
# App title
st.title('Gender Predictions using Dental Measurements')

# Load the dataset and pickle file
with open('normalized_top6_df.pkl','rb') as file:
    normalized_df=pickle.load(file)
    
# Load the trained rfc_tuned model (random_forest_classifier) 
with open('rfc_tuned_v2.pkl','rb') as file:
    rfc_model=pickle.load(file)
    
    
# Sidebar
st.sidebar.header('Input Parameters')



# Define the variables as input features
intercanine_distance_casts=st.number_input('intercanine distance casts(mm)')
right_canine_width_intraoral=st.number_input('right canine width intraoral(mm)')
right_canine_width_casts= st.number_input('right canine width casts(mm)')
left_canine_width_intraoral=st.number_input('left canine width intraoral(mm)')
left_canine_width_casts=st.number_input('left canine width casts(mm)')
left_canine_index_casts=st.number_input('left canine index casts(mm)')


#Predicting Gender based on input variables
if st.button('Predict Gender'):
    input_data= [
                  intercanine_distance_casts,
                  right_canine_width_intraoral,
                  right_canine_width_casts,
                  left_canine_width_intraoral,
                  left_canine_width_casts,
                  left_canine_index_casts]
    

    column_names=[
        'intercanine distance casts', 'right canine width intraoral',
       'right canine width casts', 'left canine width intraoral',
       'left canine width casts', 'left canine index casts']


    input_df = pd.DataFrame([input_data],columns=column_names)
    gender_pred = rfc_model.predict(input_df)
    
  # Display the predicted gender
    if gender_pred[0] == 0:
        st.write('The predicted gender is Female')
    else:
        st.write('The predicted gender is Male')  

    
    






