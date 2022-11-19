import tensorflow as tf
import torch
from transformers import BertForSequenceClassification,BertTokenizerFast
import streamlit as st
import torch.nn.functional as F
import unidecode
import nltk
import gensim
import pandas as pd
from nltk.corpus import stopwords
from PIL import Image
from pathlib import Path
import os
current_directory = Path(__file__).parent #Get current directory



@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizerFast.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    model = BertForSequenceClassification.from_pretrained("AleNunezArroyo/BETO_BolivianFN")
    return tokenizer,model

@st.cache(allow_output_mutation=True)
def get_extra():
    nltk.download("stopwords")
    stop_words = stopwords.words('spanish')
    image = open(os.path.join(current_directory, 'file/image.jpg'), 'rb')
    image = Image.open(image)
    data_test = open(os.path.join(current_directory, 'file/test_df.csv'), 'rb')
    data_test = pd.read_csv(data_test)
    df = data_test[['clean_token_head_con', 'label']]
    return stop_words, image, df


tokenizer,model = get_model()
stop_words, image, df = get_extra()

st.image(image, caption='Estudiante: Alejandro Núñez Arroyo | Tutor: Ing. Guillermo Sahonero')
col1, col2 = st.columns(2)
col1.metric("Izquierda probabilidad noticia verdadero", "0")
col2.metric("Derecha probabilidad noticia falso: ", "1")


user_input = st.text_area('Ingresar texto para revisión')
pre_pro = st.radio(
    "Seleccione el filtrado:",
    ('Con preprocesamiento', 'Sin preprocesamiento'))
st.dataframe(df)
button = st.button("Analizar")

def joined_data(text):
    try:
        text = " ".join(text)
        return (text)
    except:
        pass
    
def data_filter(text):
    result = []
    # Convierte en una lista de tokens en minúscula
    # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
    try:
        for token in gensim.utils.simple_preprocess(text):
            # En caso de que el token no esté en stop_words
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
                # remove ascents
                token = unidecode.unidecode(token)
                # transform to root word
                # token = stemmer.stem(token)
                result.append(token)
        return (joined_data(result))
    except:
        print(text)
        pass

MAX_SEQ_LEN = 21 

if user_input and button:
    if (pre_pro == 'Con preprocesamiento'):
        user_input = data_filter(user_input)
    else:
        user_input = user_input
    
    st.write("Texto de entrada al sistema: ",user_input)
    st.write("Longitud: ",len(user_input.split(' ')))
    encoded_review = tokenizer.encode_plus(
        user_input,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    model_out = model(input_ids,attention_mask)
    all_logits = torch.nn.functional.log_softmax(model_out.logits, dim=1)
    probs = F.softmax(all_logits, dim=1)
    st.write("Logits: ",probs)
    


