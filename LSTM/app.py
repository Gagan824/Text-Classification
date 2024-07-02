import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as OHE
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

ticket_data = pd.read_csv('data/ticket_data.csv')

def data_preprocess(data):
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     nltk.download('PorterStemmer')
    
    ## stop words list
    stop = stopwords.words("english")    
    stop_punc = list(set(punctuation))+stop
    
    ## Concatinating all text into one
    data['new'] = data['Title'] +" "+ data['Body']+ " "+data['Category']+" "+data['SubCategory']
    
    ## dropping off all un necessary columns
#     data = data.drop(['Title', 'Body', 'Category', 'SubCategory','Opened At'], axis = 1)
    
    ## lower casing the text
    data["lower_text"] = data['new'].map(lambda x:x.lower())
    
    def remove_stop(strings, stop_list):
        classed = [s for s in strings if s not in stop_list]
        return classed
    
    ## tokenizing the text
    data["tokenized"] = data['lower_text'].map(word_tokenize)
    
    ## removing stopwords based on stopword list
    data["selected"] = data['tokenized'].map(lambda df:remove_stop(df, stop_punc))
    
    def normalize(text):
        return " ".join(text)
    
    ## Stemmer object
    stemmer = PorterStemmer()

    ## stemming and normalizing the text
    data["stemmed"] = data['selected'].map(lambda xs:[stemmer.stem(x) for x in xs])
    data["normalized"] = data['stemmed'].apply(normalize)
    
    return data

def embeddings(data, tokenizer = None):
    ## Tokenizer object for text to vector conversion
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(data['normalized'])
        tokenized_train = tokenizer.texts_to_sequences(data['normalized'])
    else:
    ## text to vector/sequence conversion
        tokenized_train = tokenizer.texts_to_sequences(data['normalized'])

    ## adding padding if required
    train_padded = pad_sequences(tokenized_train, maxlen=15, padding="pre")
    
    return tokenizer, train_padded

def transform_x(data, tokenizer):
    output_shape = [data.shape[0],
                    data.shape[1],
                    tokenizer.word_index.keys().__len__()]
    results = np.zeros(output_shape)

    for i in range(data.shape[0]):
        for ii in range(data.shape[1]):
            results[i, ii, data[i, ii]-1] = 1
    return results

model = tf.keras.models.load_model('ticket_model.h5')
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import streamlit as st

title = st.text_input("Please Enter Title", "")

Body = st.text_input("Please Enter Description", "")

category = st.selectbox(
   "Please select category",
   ('Hardware', 'Network', 'Inquiry / Help', 'Database', 'Software' ),
   index=None,
   placeholder="Select category...",
)

sub_category = st.selectbox(
   "Please select sub category",
   ticket_data[ticket_data['Category'] == category]['SubCategory'].unique(),
   index=None,
   placeholder="Select sub category...",
)

requirement_input = {}

if title != '' and Body != '' and category != None and sub_category != None:
    
    ticket_data = data_preprocess(ticket_data)
    tokenizer, train_padded = embeddings(ticket_data)
    ## doing one hot encoding on output variable
    y_encoder = OHE().fit(np.array(ticket_data.Team).reshape(-1, 1))
    ytr_encoded = y_encoder.transform(np.array(ticket_data.Team).reshape(-1, 1)).toarray()

    requirement_input['Title'] = [title]
    requirement_input['Body'] = [Body]
    requirement_input['Category'] = [category]
    requirement_input['SubCategory'] = [sub_category]

    pred_data = pd.DataFrame(requirement_input)
    pred_data = data_preprocess(pred_data)
    tokenizer, df_padded = embeddings(pred_data, tokenizer)
    df_transformed = transform_x(df_padded, tokenizer)
    team = y_encoder.inverse_transform(model.predict(df_transformed))[0][0]

    st.divider()
    st.write(f'Team: {team}')
else:
    st.write('Please provide all information')

