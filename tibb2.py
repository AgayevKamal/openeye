import streamlit as st
import pandas as pd
import pickle
import time
import webbrowser
!pip install sklearn 
import sklearn
# Load your data
df = pd.read_csv('tibb2.csv')

# Preprocess the data
interface = st.container()

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
        user_input[col] = st.sidebar.selectbox(f' {col}', options=unique_values)
    else:
        user_input[col] = st.sidebar.number_input(
            f' {col}',
            min_value=float(df[col].min()),  # Cast to float
            max_value=float(df[col].max()),  # Cast to float
            value=float(df[col].mean()),  # Cast to float
        )

user_input_df = pd.DataFrame([user_input])

# Display the user input
st.subheader('User Input')
st.write(user_input_df)

# Predict button
with open('dolma.pickle', 'rb') as pickled_model:
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

        

# Load comments from a CSV file or create an empty DataFrame
comments_df = pd.DataFrame(columns=['User', 'Comment'])

st.write('\n\n')
st.write('\n\n')
st.write('\n\n')
st.write('\n\n')

# User input for new comment
st.subheader('Add a New Comment')
user_name=st.text_input("Submit your name")
user_comment = st.text_area('Your Comment:', '')

# Submit button
if st.button('Submit Comment'):
    # Add the new comment to the DataFrame
    new_comment = {'User': user_name, 'Comment': user_comment}
    comments_df = comments_df.append(new_comment, ignore_index=True)
    comments_df.to_csv('comments_df.csv' , index=False)
    # Display a success message
    st.success('Comment submitted successfully!')
