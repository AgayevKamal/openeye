import streamlit as st
import pandas as pd
import pickle
import time
import webbrowser
from sklearn.model_selection import train_test_split

df=pd.read_csv('tibb2.csv')
x=df.drop(columns=['target'])
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# SequentialFeatureSelector() metodundan istifadə edərək ən vacib dəyişənlərin seçilməsi üçün metodun yaradılması
classification=LogisticRegression(random_state=42)

# make_column_selector() metodu ilə təyin olunmuş data növünə əsasən sütunların seçilməsi
categoric_features = make_column_selector(dtype_include = 'object')
numeric_features = make_column_selector(dtype_include = 'number')

feature_selector = SequentialFeatureSelector(estimator = classification, scoring = 'accuracy', n_jobs = -1)
# Pipeline() metodu ilə kateqorik data növlü dəyişənlər üçün spesifik boru əməliyyatlarının yaradılması
categoric_transformer = Pipeline(steps = [('ohe', OneHotEncoder(handle_unknown = 'ignore'))])

# Pipeline() metodu ilə numerik data növlü dəyişənlər üçün spesifik boru əməliyyatlarının yaradılması
numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median'))])

# ColumnTransformer() metodu ilə sütunlara data növünə əsasən öncədən yaradılmış boru əməliyyatlarının tətbiq olunması
transformer = ColumnTransformer(transformers = [('categoric_transformer', categoric_transformer, categoric_features),
                                                ('numeric_transformer', numeric_transformer, numeric_features)], n_jobs = -1)

# Pipeline() metodu ilə boru modelinin qurulması
pipe = Pipeline(steps = [('transformer', transformer), ('classification', classification)])

pipe.fit(X = x_train, y = y_train)

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

if st.button('Predict'):
    prediction = pipe.predict(user_input_df)
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
