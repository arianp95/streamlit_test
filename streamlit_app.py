import pickle
import streamlit as st
import pandas as pd


filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))


@st.cache
def predict(sepal_lenght, sepal_width, petal_lenght, petal_width):
    prediction = model.predict(pd.DataFrame([[sepal_lenght, sepal_width, petal_lenght, petal_width]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']))
    return prediction


st.title('Iris')
st.image("images/iris.jpg")
st.header('Entre las característica de la flor:')
sepal_lenght = st.number_input('Sepal length:', min_value=0.1, max_value=6.0, value=1.0)
sepal_width = st.number_input('Sepal width:', min_value=0.1, max_value=6.0, value=1.0)
petal_lenght = st.number_input('Petal length:', min_value=0.1, max_value=6.0, value=1.0)
petal_width = st.number_input('Petal width:', min_value=0.1, max_value=6.0, value=1.0)

if st.button('Predecir tipo de flor'):
    flower = predict(sepal_lenght, sepal_width, petal_lenght, petal_width)
    if flower == 0:
        st.header('Iris setosa')
        st.image("images/setosa.png")
    elif flower == 1:
        st.header('Iris versicolor')
        st.image("images/versicolor.png")
    else:
        st.header('Iris virginica')
        st.image("images/virginica.png")
