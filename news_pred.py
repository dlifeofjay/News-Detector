import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load tokenizer and model
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

model = load_model("fake_news_bilstm_model.h5")
max_len = 500  # same value used during training

# Define prediction function
def predict_news(news_text):
    seq = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    label = "🟢 Real News" if pred > 0.5 else "🔴 Fake News"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence

# Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Type or paste a news article or headline:")
user_input = st.text_area("News Text")

if st.button("Check"):
    if user_input.strip():
        label, confidence = predict_news(user_input)
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
    else:
        st.warning("Please enter some text to analyze.")
