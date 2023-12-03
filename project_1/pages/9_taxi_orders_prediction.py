import streamlit as st
import numpy as np
import pandas as pd
import pickle 
from catboost import CatBoostRegressor
import datetime

st.set_page_config(page_title="# Прогнозирование заказов такси")

st.markdown('# Прогнозирование заказов такси')

with st.expander("Описание проекта"):
    st.write("""
        Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. 
        Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час.
        
        Описание данных:
        - datetime - время заказа
        - num_orders - число заказов
    """)

df_old = pd.read_csv('project_1/models/taxi.csv', index_col=[0], parse_dates=[0]).sort_index().resample('1H').sum()


def user_input_features():
    date = st.date_input("дата заказа такси", datetime.date(2018, 9, 6))
    time = st.time_input('время заказа такси', datetime.time(8, 45))
    target_datetime  = datetime.datetime.combine(date, time)
    data = {'datetime': target_datetime }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df = df.sort_index(axis=1)


   
def preprocessing_data(data, max_lag, rolling_mean_size, target_datetime):
    df_new = data.copy()
    df_new['month'] = df_new.index.month
    df_new['day'] = df_new.index.day
    df_new['dayofweek'] = df_new.index.dayofweek
    df_new['hour'] = df_new.index.hour
    # Создаем признаки - значения за предыдущие периоды
    for lag in range(1, max_lag + 1):
        df_new[f'lag_{lag}'] = df_new['num_orders'].shift(lag)
    # Создаем признак "скользящее среднее"
    df_new['rolling_mean'] = df_new['num_orders'].shift().rolling(rolling_mean_size).mean()
    # Удаляем пропуски
    df_new = df_new.dropna(axis=0)
    
    # Создаем DataFrame для target_datetime
    target_df = target_datetime.copy()
    target_df['month'] = target_df['datetime'].dt.month
    target_df['day'] = target_df['datetime'].dt.day
    target_df['dayofweek'] = target_df['datetime'].dt.dayofweek
    target_df['hour'] = target_df['datetime'].dt.hour
    target_df = target_df.set_index('datetime')
    
    # Создаем признаки - значения за предыдущие периоды для target_datetime
    for lag in range(1, max_lag + 1):
        target_df[f'lag_{lag}'] = df_new['num_orders'].shift(lag).iloc[-1]
    
    # Создаем признак "скользящее среднее" для target_datetime
    target_df['rolling_mean'] = df_new['num_orders'].shift().rolling(rolling_mean_size).mean().iloc[-1]

    return target_df
    
            

    
@st.cache_resource
def get_model():
    load_model = pickle.load(open('project_1/models/taxi_orders_prediction.pkl', 'rb'))
    
    return load_model

model = get_model()
target_datetime = pd.to_datetime(df['datetime'].iloc[0])
features_for_prediction = preprocessing_data(df_old, 10, 10, df)

prediction = model.predict(features_for_prediction)
st.subheader('Прогназируемое количество заказов:')
st.write(str(round(prediction[0])))