import streamlit as st
import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="# Отток клиентов «Бета-Банка»")

st.markdown('# Отток клиентов «Бета-Банка»')

with st.expander("Описание проекта:"):
    st.write(
        """Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
    Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
    Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
    Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
    Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)
    """)     

with st.expander("Описание данных:"):
    st.write("""
        Признаки:
    - CreditScore — кредитный рейтинг
    - Geography — страна проживания
    - Gender — пол
    - Age — возраст
    - Tenure — сколько лет человек является клиентом банка
    - Balance — баланс на счёте
    - NumOfProducts — количество продуктов банка, используемых клиентом
    - HasCrCard — наличие кредитной карты
    - IsActiveMember — активность клиента
    - EstimatedSalary — предполагаемая зарплата

    Целевой признак:  
    - Exited — факт ухода клиента
        """
    )

st.sidebar.header("Признаки для модели машинного обучения")

def user_input_features():
    credit_score = st.sidebar.slider('Кредитный рейтинг', 350, 850, 500)
    geography = st.sidebar.selectbox('Страна проживания', ('France', 'Spain', 'Germany'))
    gender = st.sidebar.selectbox('Пол', ('Female', 'Male'))
    age = st.sidebar.slider('Возраст', 18, 92, 25)
    tenure = st.sidebar.slider('Сколько лет человек является клиентом банка', 0, 10, 7)
    balance = st.sidebar.slider('Баланс на счёте', 0, 300000, 20000)
    num_of_products = st.sidebar.slider('Количество продуктов банка, используемых клиентом', 1, 4, 1)
    has_cr_card = st.sidebar.selectbox('Наличие кредитной карты', ('Yes', 'No'))
    is_active_member = st.sidebar.selectbox('Активность клиента', ('Yes', 'No'))
    estimated_salary = st.sidebar.slider('Предполагаемая зарплата', 0, 200000, 10000)
    
    
    data = {'credit_score': credit_score,
            'geography': geography,
            'gender': gender,
            'age': age,
            'tenure': tenure,
            'balance': balance,
            'num_of_products': num_of_products,
            'has_cr_card': has_cr_card,
            'is_active_member': is_active_member,
            'estimated_salary': estimated_salary}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Таблица с введенными вами параметрами:')
st.write(df)

def pre_category(data):
    if data == "Yes":
        return 1
    else:
        return 0

def preprocessing_data(df, scaler, ohe):
    df['has_cr_card'] = df['has_cr_card'].apply(pre_category)
    df['is_active_member'] = df['is_active_member'].apply(pre_category)
    numeric = ['credit_score', 'age', 'tenure', 'balance', 'num_of_products', 'estimated_salary']
    categorical = ['geography', 'gender']
    df[numeric] = scaler.transform(df[numeric])
    tmp = pd.DataFrame(ohe.transform(df[categorical]).toarray(), 
                                   columns=ohe.get_feature_names_out(),
                                   index=df.index)
    df.drop(categorical, axis=1, inplace=True)
    df = df.join(tmp)
    
            
    return pd.DataFrame(df, index=[0])
    
@st.cache_resource
def get_model():
    load_model = pickle.load(open('models/clients_churn.pkl', 'rb'))
    ohe_model = pickle.load(open('models/ohe_clients_churn.pkl', 'rb'))
    scaler_model = pickle.load(open('models/scaler_clients_churn.pkl', 'rb'))
    return load_model, scaler_model, ohe_model

model, sc_model, ohe_model = get_model()

df_new = preprocessing_data(df, sc_model, ohe_model)

prediction = model.predict(df_new)
prediction_proba = model.predict_proba(df_new)


st.subheader('Рекомендация')
exited = np.array(['Клиент вероятно уйдет','Клиент вероятно останется'])
st.write(exited[prediction])

st.subheader('Вероятность рекомендации')
st.write(prediction_proba)