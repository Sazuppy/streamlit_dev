import streamlit as st
import numpy as np
import pandas as pd
import pickle 
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import datetime

st.set_page_config(page_title="# Определение стоимости автомобилей", page_icon="📈")

st.markdown('# Определение стоимости автомобилей')

st.write(
    """Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. 
    В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 

Описание данных:

- DateCrawled — дата скачивания анкеты из базы
- VehicleType — тип автомобильного кузова
- RegistrationYear — год регистрации автомобиля
- Gearbox — тип коробки передач
- Power — мощность (л. с.)
- Model — модель автомобиля
- Kilometer — пробег (км)
- RegistrationMonth — месяц регистрации автомобиля
- FuelType — тип топлива
- Brand — марка автомобиля
- Repaired — была машина в ремонте или нет
- DateCreated — дата создания анкеты
- NumberOfPictures — количество фотографий автомобиля
- PostalCode — почтовый индекс владельца анкеты (пользователя)
- LastSeen — дата последней активности пользователя  

Целевой признак:  
- Price — цена (евро)
    """
)

st.sidebar.header("Признаки для модели машинного обучения")

def changes(df):
    pass

def user_input_features():
    VehicleType = st.sidebar.selectbox('тип автомобильного кузова', ('suv', 'convertible', 'sedan', 'wagon', 'small', 'bus', 'coupe',
       'unknown', 'other'))
    RegistrationYear = st.sidebar.slider('год регистрации автомобиля', 1900, 2018, 2000)
    Gearbox = st.sidebar.selectbox('тип коробки передач', ('manual', 'auto', 'unknown'))
    Power = st.sidebar.slider('мощность (л. с.)', 1, 1000, 300)
    Model = st.sidebar.selectbox('модель автомобиля', ('tiguan', 'fortwo', '3er', 'unknown', 'logan', 'mondeo', 'golf',
       'astra', 'polo', 'omega', 'zafira', 'touran', 'other', 'c_klasse',
       'cooper', '2_reihe', 'rav', 'clio', '601', '500', 'laguna', 'a4',
       'civic', 'picanto', 'combo', 'boxster', 'stilo', 'ka', 'a3', 'eos',
       '7er', 'passat', 'tt', 'focus', 'fiesta', 'twingo', 'panda',
       'e_klasse', 'xc_reihe', 'carnival', 'kuga', 'a6', 'a_klasse',
       '5er', 'caddy', '6_reihe', 'cc', 'm_klasse', 'vectra', 'mx_reihe',
       'transit', 'insignia', 'corsa', 'discovery', 'bora', 'transporter',
       'touareg', 'lupo', 'leon', 'galant', 'v50', 'vito', '1_reihe',
       'colt', 'c5', 'cl', 'c4', 'v40', '3_reihe', 'sharan', 'slk',
       'galaxy', 'z_reihe', 'kangoo', 'c_max', 'clk', 'escort',
       'scirocco', 'avensis', 'ibiza', 'alhambra', 'octavia', 'megane',
       'pajero', '1er', 'auris', 'arosa', 'roadster', 'jimny', 's_klasse',
       'punto', 'ducato', 'agila', 'a1', 'x_reihe', 'meriva', 'i_reihe',
       'seicento', 'berlingo', 'captiva', 'ceed', 'q5', '156', 'beetle',
       'fabia', '147', 'citigo', '80', '900', 'phaeton', 'sandero',
       'kalos', 'roomster', 'rx_reihe', '5_reihe', 'cordoba', 'forfour',
       'qashqai', 'a8', 's_type', 'c3', 'micra', 'matiz', 'scenic',
       'clubman', 'antara', '4_reihe', 'superb', 'santa', 'primera',
       'b_klasse', 'tigra', 'yaris', 'modus', '159', 'carisma', 'cayenne',
       'cuore', 'viano', 'x_trail', 'espace', 'exeo', 'yeti', 'fox',
       'duster', 'spider', 'grand', 'mustang', 'c2', '100', 'vivaro',
       'niva', 'corolla', 'r19', 'sorento', 'terios', 'swift', 'fusion',
       'a5', 'x_type', 'cherokee', 'one', 'verso', 'rio', 'm_reihe',
       'cr_reihe', 'altea', 'juke', 'v_klasse', 'toledo', 'jazz', 'v70',
       'delta', 'outlander', 'signum', 'jetta', 'calibra', 's60', 'doblo',
       'impreza', 'forester', '911', 'sportage', 'lybra', '850',
       'sprinter', 'sl', 'c1', 'voyager', 'kadett', 'aveo', 'bravo',
       'justy', 'almera', 'freelander', 'ptcruiser', 'tucson', 'aygo',
       'kaefer', 'up', 's_max', 'getz', 'a2', 'cx_reihe', 'elefantino',
       '90', 'lancer', 'q7', 'defender', 'ypsilon', 'c_reihe', 'accord',
       'mii', 'nubira', 'glk', 'sirion', 'lanos', 'navara', '6er',
       'croma', '300c', 'range_rover', 'g_klasse', 'range_rover_sport',
       'note', 'spark', 'b_max', 'crossfire', 'move', 'kappa', '145',
       'legacy', 'charade', 'musa', 'kalina', 'lodgy', 'serie_2', 'q3',
       'samara', 'wrangler', 'materia', 'amarok', '9000', '200', 'i3',
       'v60', 'gl', 'rangerover'))
    Kilometer = st.sidebar.slider('пробег (км)', 1000, 150000, 30000)
    FuelType = st.sidebar.selectbox('тип топлива', ('gasoline', 'petrol', 'unknown', 'electric', 'lpg', 'other', 'cng',
       'hybrid'))
    Brand = st.sidebar.selectbox('марка автомобиля', ('volkswagen', 'smart', 'bmw', 'dacia', 'ford', 'opel',
       'mitsubishi', 'mercedes_benz', 'renault', 'mini', 'peugeot',
       'toyota', 'citroen', 'trabant', 'fiat', 'audi', 'porsche', 'honda',
       'kia', 'mazda', 'volvo', 'suzuki', 'land_rover', 'seat', 'hyundai',
       'skoda', 'chevrolet', 'nissan', 'sonstige_autos', 'alfa_romeo',
       'saab', 'rover', 'daewoo', 'chrysler', 'jaguar', 'daihatsu',
       'lancia', 'jeep', 'lada', 'subaru'))
    Repaired = st.sidebar.selectbox('была машина в ремонте или нет', ('no', 'unknown', 'yes'))
    
    data = {'VehicleType': VehicleType,
            'RegistrationYear': RegistrationYear,
            'Gearbox': Gearbox,
            'Power': Power,
            'Model': Model,
            'Kilometer': Kilometer,
            'FuelType': FuelType,
            'Brand': Brand,
            'Repaired': Repaired
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df = df.sort_index(axis=1)

st.subheader('Таблица с введенными вами параметрами:')
st.write(df)
   
def preprocessing_data(df, scaler, ohe):
    numeric = ['Power', 'Kilometer', 'RegistrationYear']
    categorial = ['FuelType', 'Repaired', 'Gearbox', 'VehicleType', 'Brand', 'Model']
    df[numeric] = scaler.transform(df[numeric])
    tmp = pd.DataFrame(ohe.transform(df[categorial]).toarray(), 
                                   columns=ohe.get_feature_names_out(),
                                   index=df.index)
    df.drop(categorial, axis=1, inplace=True)
    df = df.join(tmp).sort_index(axis=1)
    
            
    return pd.DataFrame(df, index=[0])
    
@st.cache_resource
def get_model():
    load_model = pickle.load(open('project_1/models/car_cost_pred.pkl', 'rb'))
    ohe_model = pickle.load(open('project_1/models/ohe_car_cost_pred.pkl', 'rb'))
    scaler_model = pickle.load(open('project_1/models/scaler_car_cost_pred.pkl', 'rb'))
    return load_model, scaler_model, ohe_model

model, sc_model, ohe_model = get_model()

df_new = preprocessing_data(df, sc_model, ohe_model)
# st.write(df_new)
prediction = model.predict(df_new)


st.subheader('Рекомендованная стоимость')
rounded_prediction = np.around(prediction)
st.write(str(abs(rounded_prediction.item())) + ' евро')