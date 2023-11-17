from pyspark.sql import SparkSession
import streamlit as st
import numpy as np
import pandas as pd
import dill
import pyspark

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml.regression import LinearRegression

# Создание сессии Spark
spark = SparkSession.builder.appName("example").getOrCreate()

st.set_page_config(page_title="# Предсказание стоимости жилья", page_icon="📈")

st.markdown('# Предсказание стоимости жилья')

st.write(
    """В проекте вам нужно обучить модель линейной регрессии на данных о жилье в Калифорнии в 1990 году. На основе данных нужно предсказать медианную стоимость дома в жилом массиве. 
    Обучите модель и сделайте предсказания на тестовой выборке. Для оценки качества модели используйте метрики RMSE, MAE и R2.

В колонках датасета содержатся следующие данные:  
- longitude — широта;
- latitude — долгота;
- housing_median_age — медианный возраст жителей жилого массива;
- total_rooms — общее количество комнат в домах жилого массива;
- total_bedrooms — общее количество спален в домах жилого массива;
- population — количество человек, которые проживают в жилом массиве;
- households — количество домовладений в жилом массиве;
- median_income — медианный доход жителей жилого массива;
- median_house_value — медианная стоимость дома в жилом массиве;
- ocean_proximity — близость к океану.
    """
)

st.sidebar.header("Признаки для модели машинного обучения")

def changes(df):
    pass

def user_input_features():
    longitude = st.sidebar.slider('широта', -124.35, -114.31, -120.0)
    latitude = st.sidebar.slider('долгота', 32.54, 41.9, 35.0)
    housing_median_age = st.sidebar.slider('медианный возраст жителей жилого массива', 0, 70, 20)
    total_rooms = st.sidebar.slider('общее количество комнат в домах жилого массива', 0, 40000, 2000)
    total_bedrooms = st.sidebar.slider('общее количество спален в домах жилого массива', 0, 7000, 2000)
    population = st.sidebar.slider('количество человек, которые проживают в жилом массиве', 0, 40000, 5000)
    households = st.sidebar.slider('количество домовладений в жилом массиве', 1, 7000, 400)
    median_income = st.sidebar.slider('количество взрослых постояльцев', 0, 5, 40)
    ocean_proximity = st.sidebar.selectbox('близость к океану', ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))
       
    data = {'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity': ocean_proximity
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


st.subheader('Таблица с введенными вами параметрами:')
st.write(df)

st.map(df, latitude='latitude', longitude='longitude', zoom=3)
df = spark.createDataFrame(df)
    
def preprocessing_data(df, scaler, ohe, idx):
    categorical_cols = ['ocean_proximity']
    numerical_cols  = ['longitude', 'latitude', 'housing_median_age','total_rooms','total_bedrooms','population','households', 'median_income']
    df = idx.transform(df)
    df = ohe.transform(df)
    categorical_assembler = VectorAssembler(inputCols=[i + '_ohe' for i in categorical_cols], outputCol='categorical_features')
    df = categorical_assembler.transform(df)
    numerical_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")
    df = numerical_assembler.transform(df)
    df = scaler.transform(df)
    all_features = ['categorical_features','numerical_features_scaled']
    final_assembler = VectorAssembler(inputCols=all_features, outputCol='features')
    df = final_assembler.transform(df)
            
    return df
    
@st.cache_resource
def get_model():
    with open('models/model_real_estate_cost.dill', 'rb') as f:
        model = dill.load(f)
    with open('models/scaler_real_estate_cost.dill', 'rb') as f:
        sc_model = dill.load(f)
    with open('models/ohe_real_estate_cost.dill', 'rb') as f:
        ohe_model = dill.load(f)
    with open('models/idx_real_estate_cost.dill', 'rb') as f:
        idx_model = dill.load(f)
    return model, sc_model, ohe_model, idx_model

model, sc_model, ohe_model, idx_model = get_model()

df_new = preprocessing_data(df, sc_model, ohe_model, idx_model)

# prediction = model.predict(df_new)
# prediction_proba = model.predict_proba(df_new)


# st.subheader('Рекомендованная стоимость')
# rounded_prediction = np.around(prediction)
# st.write(str(abs(rounded_prediction.item())) + ' $')