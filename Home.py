import streamlit as st
from pathlib import Path
st.set_page_config(page_title="Главная страница")

from st_pages import Page, show_pages

show_pages(
    [
        Page("Home.py", "Главная страница", "🏠"),
        Page("pages/1_tariff_recommendation.py", "Рекомендация тарифа", ":receipt:"),
        Page("pages/2_clients_churn.py", "Отток клиентов «Бета-Банка»", ":classical_building:"),
        Page("pages/3_booking_cancel_pred.py", "Прогнозирование оттока клиентов в сети отелей «Как в гостях»", ":house_buildings:"),
        Page("pages/4_toxic_comments.py", "Выявление негативных комментариев с BERT", ":female-student:"),
        Page("pages/5_star_temperature.py", "Прогнозирование температуры звезды", ":star:"),
        Page("pages/6_sql_stackoverflow.py", "Анализ данных StackOverflow", ":page_facing_up:"),
        Page("pages/7_car_cost_pred.py", "Определение стоимости автомобилей", ":car:"),
        Page("pages/8_accident_prediction.py", "Разработка модели для оценки ДТП", ":rotating_light:"),
        Page("pages/9_taxi_orders_prediction.py", "Прогнозирование заказов такси", ":taxi:"),
        Page("pages/10_power_forecasting.py", "Потребление электроэнергии производством", ":factory:")
    ]
)
st.write("## Привет, Меня зовут Махнев Андрей! 👋")
st.markdown(
    """
    **О себе:** я являюсь начинающим специалистом в сфере Data Science. Начал обучение в середнине 2022г. с основ Python на курсах:
    - [Питонтьютор](https://pythontutor.ru/)
    - ["Поколение Python": курс для начинающих](https://stepik.org/course/58852/promo)
    - ["Поколение Python": курс для продвинутых](https://stepik.org/course/68343/promo)
    - ["Поколение Python": курс для профессионалов](https://stepik.org/course/82541/promo).
    - C ноября 2022г начал курс [Яндекс Практикум (2022) по направлению Data Science Plus](https://practicum.yandex.ru/data-scientist-plus/) по декабрь 2023г.   
    
    Хочу представить проект на базе [Streamlit](https://streamlit.io/) для наглядной демонтрации работы моделей машинного обучения, которые созданы на основе учебных проектов в процессе
    обучения на курсах Яндекс Практикум по направлению Data Science Plus.     
    [Ссылка на профиль с проектами на GitHub](https://github.com/Sazuppy/yandex_project)
    
    Технологии которые были мною изучены в процессе обучения и применены в данном проекте и проектах Яндекс Практикум:
    Streamlit, Keras, TensorFlow, scikit-learn, scipy, PostgreSQL, Pandas, Numpy, Matplotlib, PyTorch, Transformers.
        
    Так же хочу познакомить с отдельным проектом компьтерного зрения, который создан на основе учебного проекта Яндекс Практикум.   
    [Определние возраска человека по фотографии](https://agedetermination-xhmkruwueivfwofycnmxf7.streamlit.app/)

"""
)
