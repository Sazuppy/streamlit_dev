import streamlit as st
import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import datetime
from st_pages import show_pages_from_config
show_pages_from_config()
st.set_page_config(page_title="# Оптимизация производственных расходов металлургического комбината.")

st.markdown('# Оптимизация производственных расходов металлургического комбината.')

with st.expander("Описание проекта"):
    st.write("""
        Для оптимизации производственных расходов, металлургический комбинат решил уменьшить потребление электроэнергии на этапе обработки стали. 
        Для этого нужно контролировать температуру сплава. 
        Задача — построить модель, которая будет её предсказывать, заказчик хочет использовать разработанную модель для имитации технологического процесса.
    """)
with st.expander("Описание процесса обработки"):
    st.write("""
        Сталь обрабатывают в металлическом ковше вместимостью около 100 тонн. Чтобы ковш выдерживал высокие температуры, изнутри его облицовывают огнеупорным кирпичом. Расплавленную сталь заливают в ковш и подогревают до нужной температуры графитовыми электродами. Они установлены на крышке ковша.
Сначала происходит десульфурация — из стали выводят серу и корректируют её химический состав добавлением примесей. Затем сталь легируют — добавляют в неё куски сплава из бункера для сыпучих материалов или порошковую проволоку через специальный трайб-аппарат.
Прежде чем в первый раз ввести легирующие добавки, специалисты производят химический анализ стали и измеряют её температуру. Потом температуру на несколько минут повышают, уже после этого добавляют легирующие материалы и продувают сталь инертным газом, чтобы перемешать, а затем снова проводят измерения. Такой цикл повторяется до тех пор, пока не будут достигнуты нужный химический состав стали и оптимальная температура плавки.
Дальше расплавленная сталь отправляется на доводку металла или поступает в машину непрерывной разливки. Оттуда готовый продукт выходит в виде заготовок-слябов (англ. slab, «плита»).
    """)

with st.expander("Описание данных"):
    st.write("""
        Данные хранятся в базе данных PostgreSQL. Она состоит из нескольких таблиц:
        - steel.data_arc — данные об электродах;
        - steel.data_bulk — данные об объёме сыпучих материалов;
        - steel.data_bulk_time — данные о времени подачи сыпучих материалов;
        - steel.data_gas — данные о продувке сплава газом;
        - steel.data_temp — данные об измерениях температуры;
        - steel.data_wire — данные об объёме проволочных материалов;
        - steel.data_wire_time — данные о времени подачи проволочных материалов.

        Таблица steel.data_arc:
        - key — номер партии;
        - BeginHeat — время начала нагрева;
        - EndHeat — время окончания нагрева;
        - ActivePower — значение активной мощности;
        - ReactivePower — значение реактивной мощности.
        
        Таблица steel.data_bulk:
        - key — номер партии;
        - Bulk1 … Bulk15 — объём подаваемого материала.
        
        Таблица steel.data_bulk_time:
        - key — номер партии;
        - Bulk1 … Bulk15 — время подачи материала.
        
        Таблица steel.data_gas:
        - key — номер партии;
        - gas — объём подаваемого газа.
        
        Таблица steel.data_temp:
        - key — номер партии;
        - MesaureTime — время замера;
        - Temperature — значение температуры.
        
        Таблица steel.data_wire:
        - key — номер партии;
        - Wire1 … Wire15 — объём подаваемых проволочных материалов.
        
        Таблица steel.data_wire_time:
       -  key — номер партии;
        - Wire1 … Wire15 — время подачи проволочных материалов.
        
        Во всех файлах столбец key содержит номер партии. В таблицах может быть несколько строк с одинаковым значением key: они соответствуют разным итерациям обработки.
    """)

st.sidebar.header("Признаки для модели машинного обучения")

def changes(df):
    pass

def user_input_features():
    gas = st.sidebar.slider('объём подаваемого газа на продувку, м3/ч', 0, 100, 10)
    temp_first = st.sidebar.slider('значение температуры сплава первого замера, С', 1500, 1680, 1580)
    count = st.sidebar.slider('количество замеров температуры', 1, 20, 3)
    measure_time = st.sidebar.slider('длительность замера, с', 10, 2000, 80)
    Bulk_3 = st.sidebar.slider('объём подаваемого материала', 0, 450, 100)
    Bulk_4 = st.sidebar.slider('объём подаваемого материала', 0, 300, 100)
    Bulk_12 = st.sidebar.slider('объём подаваемого материала', 0, 2000, 500)
    Bulk_14 = st.sidebar.slider('объём подаваемого материала', 0, 650, 300)
    Bulk_15 = st.sidebar.slider('объём подаваемого материала', 0, 350, 100)
    Wire_1 = st.sidebar.slider('объём подаваемых проволочных материалов', 0.0, 22.0, 10.0)
    Wire_2 = st.sidebar.slider('объём подаваемых проволочных материалов', 0, 350, 100)
    full_power = st.sidebar.slider('полная мощность', 0.25, 21.5, 10.5)
    power_coef = st.sidebar.slider('коэффициент мощности', 0.50, 0.90, 0.60)
    
    
    
    data = {'gas': gas,
            'temp_first': temp_first,
            'count_x': count,
            'measure_time': measure_time,
            'Bulk 3': Bulk_3,
            'Bulk 4': Bulk_4,
            'Bulk 12': Bulk_12,
            'Bulk 14': Bulk_14,
            'Bulk 15': Bulk_15,
            'Wire 1': Wire_1,
            'Wire 2': Wire_2,
            'full_power': full_power,
            'power_coef': power_coef,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df = df.sort_index(axis=1)

st.subheader('Таблица с введенными вами параметрами:')
st.write(df)

def preprocessing_data(df, scaler):
    df = scaler.transform(df)
                
    return pd.DataFrame(df, index=[0])
    
@st.cache_resource
def get_model():
    load_model = pickle.load(open('project_1/models/power_forecasting.pkl', 'rb'))
    scaler_model = pickle.load(open('project_1/models/scaler_power_forecasting.pkl', 'rb'))
    return load_model, scaler_model

model, sc_model = get_model()

df_new = preprocessing_data(df, sc_model)

prediction = model.predict(df_new)


st.subheader('Температура сплава')
rounded_prediction = np.around(prediction)
st.write(str(rounded_prediction.item()))