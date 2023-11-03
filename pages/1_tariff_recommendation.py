import streamlit as st
import time
import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Рекомендация тарифов", page_icon="📈")

st.markdown('# Рекомендация тарифов')

st.write(
    """Оператор мобильной связи «Мегалайн» выяснил: многие клиенты пользуются архивными тарифами.   
    Задача состояла в построении системы, способной проанализировать поведение клиентов 
    - пользователей архивных тарифов и предложить пользователям новый тариф: «Смарт» или «Ультра».   
    Была построена модель (RandomForestClassifier) для задачи классификации, которая выберает подходящий 
    тариф с максимально большим значением accuracy (доля правильных ответов).
    
    Описание данных на которых можель была обучена:     

    * сalls — количество звонков,
    * minutes — суммарная длительность звонков в минутах,
    * messages — количество sms-сообщений,
    * mb_used — израсходованный интернет-трафик в Мб,
    * is_ultra — каким тарифом пользовался в течение месяца («Ультра» — 1, «Смарт» — 0).
    """
)

st.sidebar.header("Признаки для модели машинного обучения")

def user_input_features():
    calls = st.sidebar.slider('Количество звонков', 0, 60, 500)
    minutes = st.sidebar.slider('Количество потраченных минут', 0, 400, 3000)
    messages = st.sidebar.slider('Количество sms-сообщений', 0, 30, 500)
    mb_used = st.sidebar.slider('Количество потраченного интернет-трафика, Мб', 0, 17000, 70000)
    data = {'calls': calls,
            'minutes': minutes,
            'messages': messages,
            'mb_used': mb_used}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Таблица с введенными вами параметрами:')
st.write(df)

load_model = pickle.load(open('models/tariff_recommendation.pkl', 'rb'))

prediction = load_model.predict(df)
prediction_proba = load_model.predict_proba(df)

st.subheader('Рекомендация')
tariff = np.array(['Smart','Ultra'])
st.write(tariff[prediction])

st.subheader('Вероятность рекомендации')
st.write(prediction_proba)





