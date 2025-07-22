
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import joblib
from datetime import datetime

# -------------------------------
# ✅ Page Config
# -------------------------------
st.set_page_config(
    page_title="🐦 Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="centered",
)

# -------------------------------
# ✅ Custom CSS for Buttons & Style
# -------------------------------
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0d8ddb;
    }
    .reportview-container {
        background: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# ✅ Sidebar
# -------------------------------
with st.sidebar:
    st.title("💡 About")
    st.write("""
        This app predicts whether a tweet is **Positive** 😊 or **Negative** 😞 using a trained **Naive Bayes** model.
        
        **Built with:**  
        - 🐍 Python  
        - ✂️ NLTK  
        - ⚙️ Scikit-learn  
        - 🚀 Streamlit  
        
        ---
        📚 NLP: Text cleaning + Bag-of-Words + Naive Bayes  
        🧑‍💻 Made with ❤️ by **Hayredin M.**
    """)

# -------------------------------
# ✅ Main Title
# -------------------------------
st.markdown(
    "<h1 style='text-align: center; color: #1DA1F2;'>🐦 Twitter Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)
st.write("Enter a tweet below to see if it's **Positive** 😊 or **Negative** 😞.")

# -------------------------------
# ✅ Example Tweets
# -------------------------------
if st.checkbox("💬 Show Example Tweets"):
    st.info("""
    **Try one of these:**  
    - *I love this new phone!* 📱  
    - *This service is terrible.* 😡  
    - *What a beautiful day!* ☀️  
    """)

# -------------------------------
# ✅ Download NLTK Data
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# ✅ Load Model & Vectorizer
# -------------------------------
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# -------------------------------
# ✅ Clean Text Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# -------------------------------
# ✅ User Input
# -------------------------------
tweet = st.text_area("✏️ Your Tweet:", placeholder="Type your tweet here...")

# -------------------------------
# ✅ Analysis History Placeholder
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# ✅ Predict Button
# -------------------------------
if st.button("🔍 Analyze Sentiment"):
    if not tweet.strip():
        st.warning("⚠️ Please enter a tweet.")
    else:
        with st.spinner("Analyzing... ⏳"):
            clean = clean_text(tweet)
            vec = vectorizer.transform([clean])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = round(max(proba) * 100, 2)

        if pred == 1:
            st.success(f"✅ Sentiment: **Positive** 😊  \nConfidence: **{confidence}%**")
            st.balloons()
        else:
            st.error(f"❌ Sentiment: **Negative** 😞  \nConfidence: **{confidence}%**")

        # Save to history
        st.session_state.history.append({
            "tweet": tweet,
            "result": "Positive" if pred == 1 else "Negative",
            "confidence": confidence,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# -------------------------------
# ✅ Analysis History
# -------------------------------
if st.checkbox("🕑 Analysis History"):
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.write(f"**[{item['time']}]** — \"{item['tweet']}\" → {item['result']} ({item['confidence']}%)")
    else:
        st.info("No history yet. Analyze a tweet to see it here!")

# -------------------------------
# ✅ Feedback Section
# -------------------------------
st.markdown("---")
st.subheader("💬 Feedback")
st.write("Did you like this app? Any suggestions?")

feedback = st.text_area("Your feedback here...")
if st.button("✅ Submit Feedback"):
    if feedback.strip():
        st.success("✅ Thanks for your feedback! 🙌")
    else:
        st.warning("⚠️ Please write some feedback before submitting.")

# -------------------------------
# ✅ Footer with GitHub Link
# -------------------------------
st.markdown("""
---
📈 [View on GitHub](https://github.com/HayreKhan750/twitter-sentiment-analyzer)  
Made with ❤️ by Hayredin M.
""")
