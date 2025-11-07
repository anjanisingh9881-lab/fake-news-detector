import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ✅ Preprocess Function

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) forword in words if word not in stop_words]
    return " ".join(words)

# ✅ Load Trained Model
model = joblib.load("fake_news_model.joblib")

vectorizer = joblib.load("tfidf_vectorizer.joblib")

# ✅ UI
st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection AI")
st.write("Enter any news headline or article below:")

news = st.text_area("Paste news text here:")

# ✅ Button
if st.button("Check"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:
        # ✅ PREPROCESS
        processed = preprocess_text(news)

        # ✅ VECTORIZER
        vectorized = vectorizer.transform([processed])

        # ✅ PREDICTION
        prediction = model.predict(vectorized)[0]

        # ✅ RESULT
        st.write("Raw prediction:", prediction)

        if prediction == 1:
            st.success("✅ This news appears REAL ✅")
        else:
            st.error("❌ This news appears FAKE ❌")
