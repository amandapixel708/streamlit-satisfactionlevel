import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('cleancustomerbehaviourok.csv')
#df

X = df[['Age', 'Items Purchased', 'Spend per Item', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase']]
Y = df['Satisfaction Level']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, Y_train)
#X_test
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

#print(f'Mean Squared Error: {mse}')
#print(f'R^2 Score: {r2}')

input_data = [[29, 14, 80.014, 4.6, 1, 25]]
satisfaction_level = model.predict(input_data)

predicted_class = int(round(satisfaction_level[0]))
predicted_class = max(0, min(predicted_class, 2))

satisfaction_labels = ["Unsatisfied", "Neutral", "Satisfied"]
predicted_label = satisfaction_labels[predicted_class]

print(f'Predicted Satisfaction Level: {predicted_label}')

import pickle
import streamlit as st

# Save the model if it's not already saved
print(model)
if 'satisfaction_model' not in locals():  # Check if the model file exists
    with open('satisfaction_model.sav', 'wb') as f:
        pickle.dump(model, f)  # Assuming 'model' is your trained RandomForestRegressor

# load model
satisfaction_model = pickle.load(open('satisfaction_model.sav', 'rb'))

# Judul Web
st.title('Customer Satisfaction Level')

# Membagi kolom
col1, col2 = st.columns(2)

with col1 :
    Age = st.text_input('Age')

with col2 :
    Items_Purchased = st.text_input('Amount of Items Purchased')

with col1 :
    Spend_per_Item = st.text_input('Spend per Item, Example 77.910')

with col2 :
    Average_Rating = st.text_input('Average Rating Range 1-5, Example 3.5')

with col1 :
    Discount_Applied = st.text_input('Discount Applied (0 = No, 1 = Yes)')

with col2 :
    DSLP = st.text_input('Days Since Last Purchase')

sat_diagnosis = ''
if st.button('Predict'):
    sat_prediction = satisfaction_model.predict([[Age,Items_Purchased,Spend_per_Item,
                                              Average_Rating,Discount_Applied,DSLP]])
    if(sat_prediction[0] == 0):
        sat_diagnosis = 'Customer is Unsatisfied'
    elif (sat_prediction[0] == 1):
        sat_diagnosis = 'Customer is Neutral'
    else:
        sat_diagnosis = 'Customer is Satisfied'

    st.success(sat_diagnosis)


