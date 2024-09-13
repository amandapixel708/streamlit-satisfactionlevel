import numpy as np
import pandas as pd
import statsmodels
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
#print(X)
#print(Y)
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

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load trained model
satisfaction_model = pickle.load(open('satisfaction_model.sav', 'rb'))

# Judul Web
st.title('Customer Satisfaction Level Prediction')

# Membagi kolom
col1, col2 = st.columns(2)

with col1:
    Age = st.text_input('Age', '0')

with col2:
    Items_Purchased = st.text_input('Amount of Items Purchased', '0')

with col1:
    Spend_per_Item = st.text_input('Spend per Item in $, Example 77.91', '0')

with col2:
    Average_Rating = st.text_input('Average Rating Range 1-5, Example 3.5', '0')

with col1:
    Discount_Applied = st.text_input('Discount Applied (0 = No, 1 = Yes)', '0')

with col2:
    DSLP = st.text_input('Days Since Last Purchase', '0')

# Logika untuk prediksi
sat_diagnosis = ''
if st.button('Predict'):
    try:
        # Convert inputs to proper numeric values
        Age = float(Age)
        Items_Purchased = float(Items_Purchased)
        Spend_per_Item = float(Spend_per_Item)
        Average_Rating = float(Average_Rating)
        Discount_Applied = int(Discount_Applied)
        DSLP = float(DSLP)

        # Predict using the loaded model
        sat_prediction = satisfaction_model.predict([[Age, Items_Purchased, Spend_per_Item,
                                                      Average_Rating, Discount_Applied, DSLP]])

        # Convert numerical prediction to the nearest integer class (0, 1, 2)
        predicted_class = int(round(sat_prediction[0]))
        predicted_class = max(0, min(predicted_class, 2))  # Ensure the prediction is within valid range

        # Define satisfaction labels
        satisfaction_labels = ["Unsatisfied", "Neutral", "Satisfied"]
        sat_diagnosis = satisfaction_labels[predicted_class]

        st.success(f'Predicted Satisfaction Level: {sat_diagnosis}')

    except ValueError:
        st.error('Please enter valid numerical inputs.')
