import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

model = joblib.load("fake_news_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection AI")
st.write("Enter any news headline or article below:")

news = st.text_area("Paste news text here:")

if st.button("Check"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        processed = preprocess_text(news)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]

        # Debug line
        st.write("Raw prediction:", prediction)

        if prediction == 1:
            st.success("✅ This news appears REAL ✅")
        else:
            st.error("❌ This news appears FAKE ❌")
