import streamlit as st
import pickle as pl
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pandas as pd
#nltk.download('all')

tfidf = pl.load(open("vectorizer.pkl",'rb'))
model = pl.load(open('model.pkl','rb'))

# Load the slang dictionary
try:
    slangs = pd.read_csv('C://Users//Public//smsClassifier//slang.csv')
    slangs.drop('Unnamed: 0', axis=1, inplace=True)
    chat_words = slangs.set_index('acronym')['expansion'].to_dict()
except FileNotFoundError as e:
    st.error(f"Error loading slang.csv: {e}")
    st.stop()

# Initialize PorterStemmer
ps = PorterStemmer()

# Define the text transformation function
def transform_text(text):
    # Normalize case
    text = text.lower()

    # Replace chat abbreviations
    def replace_chat_words(word):
        return chat_words.get(word, word)

    text = " ".join(replace_chat_words(w) for w in text.split())

    # Tokenize and clean
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words() and word not in string.punctuation]
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Streamlit UI
st.title('Email/SMS Spam Classifier')

input_text = st.text_area('Enter your text:')

if st.button('PREDICT'):
    # Preprocess input
    transformed = transform_text(input_text)

    # Vectorize
    vectorized_text = tfidf.transform([transformed])
    #print(vectorized_text)

    # Predict
    result = model.predict(vectorized_text)[0]

    # Display result
    if result == 0:
        st.header('NOT SPAM')
    else:
        st.header('SPAM')



