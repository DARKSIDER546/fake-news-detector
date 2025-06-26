import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")
user_input = st.text_area("Enter the News Article/Text")

if st.button("Predict"):
    if user_input:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        if prediction[0] == "FAKE":
            st.error("❌ This news is likely FAKE!")
        else:
            st.success("✅ This news appears REAL.")
    else:
        st.warning("Please enter some news content.")
