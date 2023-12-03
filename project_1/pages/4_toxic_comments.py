import streamlit as st
import pandas as pd
import pickle
import transformers as tfs
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch as t
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from st_pages import show_pages_from_config
show_pages_from_config()
st.set_page_config(page_title="# Выявление негативных комментариев с BERT")

st.markdown('# Выявление негативных комментариев с BERT')

with st.expander("Описание проекта"):
    st.write("""
        Интернет-магазин «Викишоп» запускает новый сервис. 
        Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. 
        То есть клиенты предлагают свои правки и комментируют изменения других. 
        Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
    """)


def detect_language(text):
    first_letter = text[0].lower()
    if 'а' <= first_letter <= 'я':
        return 'ru'
    
    
def ru_bert_comments(text):

    model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if t.cuda.is_available():
        model.cuda()
        
    def text2toxicity(text, aggregate=True):
        """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
        with t.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
            proba = t.sigmoid(model(**inputs).logits).cpu().numpy()
        if isinstance(text, str):
            proba = proba[0]
        if aggregate:
            return 1 - proba.T[0] * (1 - proba.T[-1])
        return proba
   
    return round(text2toxicity(text, True))

def preprocessing(text):
    lang = detect_language(text)
    if lang == "ru":
        return ru_bert_comments(str(text))
    else:
        tokenizer = tfs.AutoTokenizer.from_pretrained('unitary/toxic-bert')
        model = tfs.AutoModel.from_pretrained('unitary/toxic-bert')
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    
    if len(tokenized) > 512:
        truncated_tokens = tokenized[:510]  
        truncated_tokens = [101] + truncated_tokens + [102]
        tokenized = truncated_tokens
    
    padded = np.array(tokenized + [0] * (512 - len(tokenized)))
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = t.tensor(padded)
    attention_mask = t.tensor(attention_mask)

    with t.no_grad():
        embeddings = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))[0][:, 0, :].cpu().numpy()

    return embeddings

def query(features):
    model = pickle.load(open('project_1/models/toxic_comments_bert.pkl', 'rb'))
    predict = model.predict(features)
    return predict

comment = st.text_area("Введите ваш комментарий, модель работает на английском и русском языках и нажмите Ctrl+Enter", "")
result = None

if comment:
    st.markdown('## Результат:')
    embeddings = preprocessing(comment)
    if isinstance(embeddings, int):
        if embeddings == 0:
            result = 'Комментарий не токсичный'
        else:
            result = 'Комментарий является токсичным'
    else:        
        if query(embeddings) == 0:
            result = 'Комментарий не токсичный'
        else:
            result = 'Комментарий является токсичным'
        
if result is not None:
    st.write(result)
    
    
    
        




