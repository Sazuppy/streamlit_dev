import streamlit as st
import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import datetime
from st_pages import show_pages_from_config

st.set_page_config(page_title="# Разработка модели для оценки ДТП по выбранному маршруту движения.")

st.markdown('# Разработка модели для оценки ДТП по выбранному маршруту движения.')

st.write(
    """Цель:   
Нужно создать систему, которая могла бы оценить риск ДТП по выбранному маршруту движения. 
Под риском понимается вероятность ДТП с любым повреждением транспортного средства. 
Как только водитель забронировал автомобиль, сел за руль и выбрал маршрут, система должна оценить уровень риска. 
Если уровень риска высок, водитель увидит предупреждение и рекомендации по маршруту.

    """
)

st.sidebar.header("Признаки для модели машинного обучения")

def changes(df):
    pass

def user_input_features():
    meal = st.sidebar.selectbox('опции заказа', ('BB', 'FB', 'HB', 'SC'))
    country = st.sidebar.selectbox('гражданство постояльца', ('GBR', 'PRT', 'ESP', 'IRL', 'FRA', 'Others', 'USA', 'DEU', 'BEL', 'CHE', 'NLD', 'ITA', 'BRA', 'AUT'))
    distribution_channel = st.sidebar.selectbox('канал дистрибуции заказа', ('Direct', 'TA/TO', 'Corporate', 'GDS'))
    reserved_room_type = st.sidebar.selectbox('тип забронированной комнаты', ('A', 'C', 'D', 'E', 'G', 'F', 'B'))
    customer_type = st.sidebar.selectbox('тип заказчика', ('Transient', 'Contract', 'Transient-Party', 'Group'))
    adults = st.sidebar.slider('количество взрослых постояльцев', 0, 6, 2)
    children = st.sidebar.slider('количество детей от 3 до 14 лет', 0, 5, 2)
    babies = st.sidebar.slider('количество младенцев', 0, 1, 5)
    days_in_waiting_list = st.sidebar.slider('сколько дней заказ ожидал подтверждения', 0, 250, 0)
    previous_cancellations = st.sidebar.slider('количество отменённых заказов у клиента', 0, 30, 0)
    data_lead = st.sidebar.date_input("день бронирования", datetime.date(2019, 7, 6))
    end_time = st.sidebar.date_input("день заезда", datetime.date(2019, 7, 20))
    count_day = st.sidebar.slider('Количество дней проживания', 0, 31, 0)
    data_back = end_time + datetime.timedelta(days=count_day)
    lead_time = (end_time-data_lead).days
    total_of_special_requests = st.sidebar.slider('количество специальных отметок', 0, 8, 0)
    arrival_date_day_of_month = end_time.day
    arrival_date_year = end_time.year
    arrival_date_month = end_time.month
    arrival_date_week_number = end_time.isocalendar()[1]
    
    stays_in_weekend_nights = 0
    stays_in_week_nights = 0
    total_nights = stays_in_weekend_nights + stays_in_week_nights
    current_date = end_time
    while current_date < data_back:
        if current_date.weekday() < 5:  # Понедельник (0) - Пятница (4)
            stays_in_week_nights += 1
        else:
            stays_in_weekend_nights += 1
        current_date += datetime.timedelta(days=1)
    
    is_repeated_guest = st.sidebar.selectbox('признак того, что гость бронирует номер второй раз', ('Yes', 'No'))
    previous_bookings_not_canceled = st.sidebar.slider('количество подтверждённых заказов у клиента', 0, 60, 0)
    required_car_parking_spaces = st.sidebar.selectbox('необходимость места для автомобиля', ('Yes', 'No'))
    booking_changes = st.sidebar.slider('количество измененных вами параметров', 0, 10, 0)
    
    
    
    data = {'meal': meal,
            'country': country,
            'distribution_channel': distribution_channel,
            'reserved_room_type': reserved_room_type,
            'customer_type': customer_type,
            'lead_time': lead_time,
            'adults': adults,
            'children': children,
            'booking_changes': booking_changes,
            'babies': babies,
            'days_in_waiting_list': days_in_waiting_list,
            'previous_cancellations': previous_cancellations,
            'total_nights': total_nights,
            'total_of_special_requests': total_of_special_requests,
            'arrival_date_day_of_month': arrival_date_day_of_month,
            'arrival_date_year': arrival_date_year,
            'arrival_date_month': arrival_date_month,
            'arrival_date_week_number': arrival_date_week_number,
            'stays_in_weekend_nights': stays_in_weekend_nights,
            'stays_in_week_nights': stays_in_week_nights,
            'is_repeated_guest': is_repeated_guest,
            'previous_bookings_not_canceled': previous_bookings_not_canceled,
            'required_car_parking_spaces': required_car_parking_spaces,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df = df.sort_index(axis=1)

st.subheader('Таблица с введенными вами параметрами:')
# st.write(df)

def pre_category(data):
    if data == "Yes":
        return 1
    else:
        return 0
    
def preprocessing_data(df, scaler, ohe):
    df['is_repeated_guest'] = df['is_repeated_guest'].apply(pre_category)
    df['required_car_parking_spaces'] = df['required_car_parking_spaces'].apply(pre_category)
    numeric = ['adults', 'children', 'booking_changes', 'babies', 'days_in_waiting_list', 'previous_cancellations', 'lead_time',
    'total_nights', 'total_of_special_requests', 'arrival_date_day_of_month', 'arrival_date_year', 'arrival_date_month', 
    'arrival_date_week_number', 'stays_in_weekend_nights', 'stays_in_week_nights', 'is_repeated_guest', 'previous_bookings_not_canceled',
    'required_car_parking_spaces', 'booking_changes']
    categorical = ['meal', 'country', 'distribution_channel', 'reserved_room_type', 'customer_type']
    df[numeric] = scaler.transform(df[numeric])
    tmp = pd.DataFrame(ohe.transform(df[categorical]).toarray(), 
                                   columns=ohe.get_feature_names_out(),
                                   index=df.index)
    df.drop(categorical, axis=1, inplace=True)
    df = df.join(tmp).sort_index(axis=1)
    
            
    return pd.DataFrame(df, index=[0])
    
@st.cache_resource
def get_model():
    load_model = pickle.load(open('project_1/models/booking_cancel_pred.pkl', 'rb'))
    ohe_model = pickle.load(open('project_1/models/ohe_booking_cancel_pred.pkl', 'rb'))
    scaler_model = pickle.load(open('project_1/models/scaler_booking_cancel_pred.pkl', 'rb'))
    return load_model, scaler_model, ohe_model

model, sc_model, ohe_model = get_model()

df_new = preprocessing_data(df, sc_model, ohe_model)
# st.write(df_new)
prediction = model.predict(df_new)
prediction_proba = model.predict_proba(df_new)


st.subheader('Рекомендация')
exited = np.array(['Клиент вероятно оставит бронь','Клиент вероятно отменит бронь'])
st.write(exited[prediction])

st.subheader('Вероятность рекомендации')
st.write(prediction_proba)