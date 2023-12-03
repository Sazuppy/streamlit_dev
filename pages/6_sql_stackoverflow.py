import streamlit as st
import pandas as pd
from sqlalchemy import create_engine 
from PIL import Image

st.set_page_config(page_title="# Анализ данных StackOverflow")

st.markdown('# Анализ данных StackOverflow')

with st.expander("Описание проекта"):
    st.write("""
        Вы будете работать с базой данных StackOverflow — сервиса вопросов и ответов о программировании. 
    StackOverflow похож на социальную сеть — пользователи сервиса задают вопросы, отвечают на посты, оставляют комментарии и ставят оценки другим ответам.  
Вы будете работать с версией базы, где хранятся данные о постах за 2008 год, но в таблицах вы найдёте информацию и о более поздних оценках, которые эти посты получили.  

    Описание данных:

- Таблица badges:
    Хранит информацию о значках, которые присуждаются за разные достижения. Например, пользователь, правильно ответивший на большое количество вопросов про PostgreSQL, может получить значок postgresql. 
    - id	Идентификатор значка, первичный ключ таблицы
    - name	Название значка
    - user_id	Идентификатор пользователя, которому присвоили значок, внешний ключ, отсылающий к таблице users
    - creation_date	Дата присвоения значка
    
- Таблица post_types:
Содержит информацию о типе постов. Их может быть два:
    - Question — пост с вопросом;
    - Answer — пост с ответом.

    - id	Идентификатор поста, первичный ключ таблицы
    - type	Тип поста
- Таблица posts:
Содержит информацию о постах.

    - id	Идентификатор поста, первичный ключ таблицы
    - title	Заголовок поста
    - creation_date	Дата создания поста
    - favorites_count	Число, которое показывает, сколько раз пост добавили в «Закладки»
    - last_activity_date	Дата последнего действия в посте, например комментария
    - last_edit_date	Дата последнего изменения поста
    - user_id	Идентификатор пользователя, который создал пост, внешний ключ к таблице users
    - parent_id	Если пост написали в ответ на другую публикацию, в это поле попадёт идентификатор поста с вопросом
    - post_type_id	Идентификатор типа поста, внешний ключ к таблице post_types
    - score	Количество очков, которое набрал пост
    - views_count	Количество просмотров
- Таблица users:
Содержит информацию о пользователях.

    - id	Идентификатор пользователя, первичный ключ таблицы
    - creation_date	Дата регистрации пользователя
    - display_name	Имя пользователя
    - last_access_date	Дата последнего входа
    - location	Местоположение
    - reputation	Очки репутации, которые получают за хорошие вопросы и полезные ответы
    - views	Число просмотров профиля пользователя
- Таблица vote_types:
Содержит информацию о типах голосов. Голос — это метка, которую пользователи ставят посту. Типов бывает несколько: 
    - UpMod — такую отметку получают посты с вопросами или ответами, которые пользователи посчитали уместными и полезными.
    - DownMod — такую отметку получают посты, которые показались пользователям наименее полезными.
    - Close — такую метку ставят опытные пользователи сервиса, если заданный вопрос нужно доработать или он вообще не подходит для платформы.
    - Offensive — такую метку могут поставить, если пользователь ответил на вопрос в грубой и оскорбительной манере, например, указав на неопытность автора поста.
    - Spam — такую метку ставят в случае, если пост пользователя выглядит откровенной рекламой.

    - id	Идентификатор типа голоса, первичный ключ
    - name	Название метки
- Таблица votes:
    Содержит информацию о голосах за посты. 
    - id	Идентификатор голоса, первичный ключ
    - post_id	Идентификатор поста, внешний ключ к таблице posts
    - user_id	Идентификатор пользователя, который поставил посту голос, внешний ключ к таблице users
    - bounty_amount	Сумма вознаграждения, которое назначают, чтобы привлечь внимание к посту
    - vote_type_id	Идентификатор типа голоса, внешний ключ к таблице vote_types
    - creation_date	Дата назначения голоса
    """)



db_config = {
    'user': 'praktikum_student', # имя пользователя
    'pwd': 'Sdf4$2;d-d30pp', # пароль
    'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
    'port': 6432, # порт подключения
    'db': 'data-analyst-advanced-sql' # название базы данных
    }  

connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
        db_config['user'],
        db_config['pwd'],
        db_config['host'],
        db_config['port'],
        db_config['db'],
    )
engine = create_engine(connection_string)

def query_db(query):
    return pd.read_sql_query(query, con=engine)

query_1 = '''SELECT date_trunc('month', creation_date) as month_date, sum(views_count) as total_views
FROM stackoverflow.posts
WHERE extract( YEAR from creation_date) = '2008'
GROUP BY month_date
ORDER BY total_views DESC
'''
query_2 = '''SELECT u.display_name, count(DISTINCT p.user_id)
FROM stackoverflow.users as u
JOIN stackoverflow.posts as p ON p.user_id = u.id
JOIN stackoverflow.post_types as pt ON pt.id = p.post_type_id
WHERE pt.type = 'Answer' AND
p.creation_date::date BETWEEN u.creation_date AND (u.creation_date::date + INTERVAL '1 month')
GROUP BY u.display_name
HAVING count(p.user_id)>100
ORDER BY u.display_name
'''
query_3 = '''WITH  dt as (SELECT u.id
FROM stackoverflow.posts as p
JOIN  stackoverflow.users as u ON p.user_id = u.id
WHERE DATE_TRUNC('month', u.creation_date) = '2008-09-01' AND
DATE_TRUNC('month', p.creation_date) = '2008-12-01')
SELECT date_trunc('month', p.creation_date)::date as month, count(p.id)
FROM stackoverflow.posts as p
WHERE p.user_id in (SELECT * FROM dt) AND
EXTRACT(YEAR FROM p.creation_date) = '2008'
GROUP BY month
ORDER BY month DESC
'''
query_4 = '''SELECT user_id, AVG(avg_daily)
FROM (SELECT DISTINCT user_id, date_trunc('day', creation_date)::date as t,
count(id) OVER (PARTITION BY user_id, date_trunc('day', creation_date)::date) as avg_daily,
count(id) OVER (PARTITION BY user_id, date_trunc('month', creation_date)::date) as cnt
FROM stackoverflow.posts
WHERE date_trunc('month', creation_date)::date = '2008-08-01') as dt
WHERE cnt>120
GROUP BY user_id
ORDER BY AVG(avg_daily)
'''

examples = {'Выводит общую сумму просмотров постов за каждый месяц 2008 года':query_1,
            'Выводит имена самых активных пользователей, которые в первый месяц после регистрации (включая день регистрации) дали больше 100 ответов':query_2,
            'Выводит количество постов за 2008 год по месяцам. Отбирает посты от пользователей, которые зарегистрировались в сентябре 2008 года и сделали хотя бы один пост в декабре того же года.':query_3,
            'Найдет среднее количество постов пользователей в день за август 2008 года. Отберет данные о пользователях, которые опубликовали больше 120 постов за август. Дни без публикаций не учитывает.':query_4,
            }
with st.expander("Схема быза данных"):
    image = Image.open('image/Frame.png')
    st.image(image)
    
with st.expander("Примеры SQL запросов"):
    
    query = st.selectbox('Выберете один из запросов:', ('Выводит общую сумму просмотров постов за каждый месяц 2008 года',
                                                        'Выводит имена самых активных пользователей, которые в первый месяц после регистрации (включая день регистрации) дали больше 100 ответов',
                                                        'Выводит количество постов за 2008 год по месяцам. Отбирает посты от пользователей, которые зарегистрировались в сентябре 2008 года и сделали хотя бы один пост в декабре того же года.',
                                                        'Найдет среднее количество постов пользователей в день за август 2008 года. Отберет данные о пользователях, которые опубликовали больше 120 постов за август. Дни без публикаций не учитывает.',
                                                        ))
    
    st.code(examples[query], language="sql", line_numbers=False)
    if st.button("Запуск запроса"):
        st.markdown('## Результат запроса:')
        st.write(query_db(examples[query]))



def submit():
    st.session_state.title = st.session_state.widget
    st.session_state.widget = ""

st.text_input("Введите ваш SQL запрос", key="widget", on_change=submit)

if 'title' not in st.session_state:
    st.session_state.title = ""    

title = st.session_state.title    



# st.write(title)       
# title = st.text_input('Введите ваш SQL запрос')
        
if title:
    st.markdown('## Ваш запрос:')
    st.code(title, language="sql", line_numbers=False)
    st.markdown('## Результат запроса:')
    try:
        st.write(query_db(title))
    except:
        'Запрос неверен, убедитесь в правильности запроса'
    
    
        




