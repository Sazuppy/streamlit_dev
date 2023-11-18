import streamlit as st
import numpy as np
import pandas as pd
import pickle 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import ceil
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="# Прогнозирование температуры звезды", page_icon="📈")

st.markdown('# Прогнозирование температуры звезды')

with st.expander("Описание проекта"):
    st.write("""
        Вам пришла задача от обсерватории «Небо на ладони»: придумать, как с помощью нейросети определять температуру на поверхности обнаруженных звёзд. Обычно для расчёта температуры учёные пользуются следующими методами:  
- Закон смещения Вина.
- Закон Стефана-Больцмана.
- Спектральный анализ.  

Каждый из них имеет плюсы и минусы. Обсерватория хочет внедрить технологии машинного обучения для предсказания температуры звёзд, надеясь, что этот метод будет наиболее точным и удобным.  
В базе обсерватории есть характеристики уже изученных 240 звёзд.
**Характеристики**   
- Относительная светимость L/Lo — светимость звезды относительно Солнца.
- Относительный радиус R/Ro — радиус звезды относительно радиуса Солнца.
- Абсолютная звёздная величина Mv — физическая величина, характеризующая блеск звезды.
- Звёздный цвет (white, red, blue, yellow, yellow-orange и др.) — цвет звезды, который определяют на основе спектрального анализа.
- Тип звезды.  
    - 0 - Коричневый карлик
    - 1 - Красный карлик
    - 2 - Белый карлик
    - 3 - Звёзды главной последовательности
    - 4 - Сверхгигант
    - 5 - Гипергигант  

- Абсолютная температура T(K) — температура на поверхности звезды в Кельвинах.  

В этом самостоятельном проекте вам необходимо разработать нейронную сеть, которая поможет предсказывать абсолютную температуру на поверхности звезды.
 Справочная информация:  
Светимость Солнца (англ. Average Luminosity of Sun)  
 $L_0 = 3.828 \cdot 10^{26}\,Вт$   

Радиус Солнца (англ. Average Radius of Sun)   
 $R_0 = 6.9551\cdot 10^8\,м$  
    """)


st.sidebar.header("Признаки для модели машинного обучения")

def star_type_cat(type_star):
    type_dict = {'Коричневый карлик':0,
    'Красный карлик':1,
    'Белый карлик':2,
    'Звёзды главной последовательности':3,
    'Сверхгигант':4,
    'Гипергигант':5 
    }
    return type_dict[type_star]

def user_input_features():
    star_color = st.sidebar.selectbox('цвет звезды, который определяют на основе спектрального анализа', ('red', 'blue', 'white', 'blue-white', 'orange', 'yellow-white', 'whitish'))
    luminosity = st.sidebar.slider('светимость звезды относительно Солнца', 0.00008, 900000.0, 2000.0)
    radius = st.sidebar.slider('радиус звезды относительно радиуса Солнца', 0.007, 2000.0, 200.0)
    abs_magnitude = st.sidebar.slider('физическая величина, характеризующая блеск звезды', -12.0, 25.0, 10.0)
    star_type = st.sidebar.selectbox('цвет звезды, который определяют на основе спектрального анализа', ('Коричневый карлик', 'Красный карлик', 'Белый карлик', 'Звёзды главной последовательности',
                                                                                                         'Сверхгигант', 'Гипергигант'))
    
    data = {'luminosity': luminosity,
            'radius':radius,
            'abs_magnitude':abs_magnitude,
            'star_color':star_color,
            'star_type':star_type
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df = df.sort_index(axis=1)

st.subheader('Таблица с введенными вами параметрами:')
st.write(df)
   
def preprocessing_data(df, scaler, ohe):
    df['star_type']=df['star_type'].apply(star_type_cat)
    numeric = ['luminosity', 'radius', 'abs_magnitude']
    categorial = ['star_color', 'star_type']
    df[numeric] = scaler.transform(df[numeric])
    tmp = pd.DataFrame(ohe.transform(df[categorial]).toarray(), 
                                   columns=ohe.get_feature_names_out(),
                                   index=df.index)
    df.drop(categorial, axis=1, inplace=True)
    df = df.join(tmp).sort_index(axis=1)
    df = torch.FloatTensor(df.values)
    return df

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.act1 = nn.Tanh()
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.act2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
        
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        
        out = self.fc2(out)
        out = self.act2(out)
        
        out = self.fc3(out)
        
        return out    


def get_model_pre():
    ohe_model = pickle.load(open('project_1/models/ohe_star_temperature_pred.pkl', 'rb'))
    scaler_model = pickle.load(open('project_1/models/scaler_star_temperature_pred.pkl', 'rb'))
    return scaler_model, ohe_model

def get_model():
    net = Net(df_new.shape[1], 700, 850, 1)
    net.load_state_dict(torch.load('project_1/models/star_temperature_pred.pkl'))
    net.eval()
    prediction = net.forward(df_new).detach().numpy()[0][0]
    
    return prediction

sc_model, ohe_model = get_model_pre()
df_new = preprocessing_data(df, sc_model, ohe_model)

model_pred = get_model()

st.subheader('Температура звезды:')
st.write(str(model_pred) + ' K')


