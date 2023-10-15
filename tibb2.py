import streamlit as st
import pandas as pd
import pickle
import time
import webbrowser
from sklearn.linear_model import LogisticRegression

# Load your data
df = pd.read_csv('tibb.csv')

# Preprocess the data
from sklearn.preprocessing import LabelEncoder

# Initialize session state
if 'button_visible' not in st.session_state:
    st.session_state.button_visible = False

# Streamlit app
st.title('Medical Diagnosis App')

# User input
st.sidebar.header('User Input')
user_input = {}
for col in df.columns[:-1]:  # Exclude the target column
    if df[col].dtype == 'O':  # Check if the column is categorical
        unique_values = df[col].unique()
        user_input[col] = st.sidebar.selectbox(f'Enter {col}', options=unique_values)
    else:
        user_input[col] = st.sidebar.number_input(
            f'Enter {col}',
            min_value=float(df[col].min()),  # Cast to float
            max_value=float(df[col].max()),  # Cast to float
            value=float(df[col].mean()),  # Cast to float
        )

user_input_df = pd.DataFrame([user_input])

# Display the user input
st.subheader('User Input')
st.write(user_input_df)

# Predict button
with open('pipe.pickle', 'rb') as pickled_model:
    model = pickle.load(pickled_model)

if st.button('Predict'):
    prediction = model.predict(user_input_df)
    with st.spinner('Getting diagnostic result...'):
        time.sleep(1)
    st.markdown(f'### Diagnostic result:  {prediction}')

    # Set button visibility to True after predicting
    st.session_state.button_visible = True

# Learn More button
if st.session_state.button_visible:
    if st.button('Səhhətinizlə bağlı dərmanları linkdən keçid edərək əldə edə bilərsiniz!'):
        webbrowser.open('https://aptekonline.az/')
