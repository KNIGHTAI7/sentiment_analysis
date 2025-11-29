import streamlit as st
import joblib



# Load model and vectorizer
model = joblib.load("model/emotion_svm_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

# Emotion labels (must match your dataset labels)
emotion_labels = ["anger", "fear", "joy", "love", "sad", "suprise"]

# UI
st.title("🧠 Sentiment Emotion Predictor")
st.write("Type a sentence and the model will predict the emotion.")

text_input = st.text_area("Enter a sentence")

if st.button("Predict Emotion"):
    if text_input.strip():
        vectorized_text = vectorizer.transform([text_input])
        prediction = model.predict(vectorized_text)[0]
        st.success(f"Predicted Emotion: **{prediction.upper()}**")
    else:
        st.warning("Please enter a sentence.")
