import re
import json
import numpy as np
import streamlit as st
import urllib.request
import tf_keras as keras
from tf_keras.preprocessing import sequence
from tf_keras.models import load_model

# Download IMDB word index directly — no tensorflow needed
@st.cache_resource
def load_word_index():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json"
    with urllib.request.urlopen(url) as f:
        word_index = json.loads(f.read().decode())
    return word_index

word_index = load_word_index()

# Load saved model
@st.cache_resource
def load_saved_model():
    return load_model('imdb_lstm_model.h5')

model = load_saved_model()

# Preprocess user input
def preprocess_review(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    encoded_review = []
    for word in words:
        idx = word_index.get(word, 2)
        encoded_review.append(min(idx + 3, 9999))
    padded = sequence.pad_sequences(
        [encoded_review], maxlen=256, padding='post', truncating='post'
    )
    return padded

# Prediction function
def predict_sentiment(review):
    preprocessed = preprocess_review(review)
    score = model.predict(preprocessed, verbose=0)[0][0]
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, float(score)

# Streamlit app
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

user_review = st.text_area("Enter your movie review:")

if st.button("Classify"):
    if user_review:
        with st.spinner("Analysing..."):
            sentiment, score = predict_sentiment(user_review)
        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment} 😊")
        else:
            st.error(f"Predicted Sentiment: {sentiment} 😞")
        st.write(f"**Confidence Score:** {score:.4f}")
        st.progress(score if sentiment == "Positive" else 1 - score)
    else:
        st.warning("Please enter a movie review to classify.")
