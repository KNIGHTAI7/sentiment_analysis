import joblib

# -----------------------
# LOAD MODEL & TF-IDF
# -----------------------
model = joblib.load(r"D:\PRIYANSHU(P)\sentiment_analysis\model\emotion_svm_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict_emotion(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# -----------------------
# USER INPUT TEST
# -----------------------
if __name__ == "__main__":
    while True:
        user_text = input("\nEnter a sentence (or type 'exit' to quit): ")

        if user_text.lower() == "exit":
            print("👋 Goodbye!")
            break

        emotion = predict_emotion(user_text)
        print(f"➡ Emotion detected: {emotion}")
