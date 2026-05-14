import re
import numpy as np
import streamlit as st
import tf_keras as keras
from tf_keras.preprocessing import sequence
from tf_keras.models import load_model
from tensorflow.keras.datasets import imdb

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load saved model
model = load_model('imdb_lstm_model.h5')

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
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

user_review = st.text_area("Enter your movie review:")

if st.button("Classify"):
    if user_review:
        sentiment, score = predict_sentiment(user_review)
        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment} 😊")
        else:
            st.error(f"Predicted Sentiment: {sentiment} 😞")
        st.write(f"**Confidence Score:** {score:.4f}")
        st.progress(score if sentiment == "Positive" else 1 - score)
    else:
        st.warning("Please enter a movie review to classify.")
