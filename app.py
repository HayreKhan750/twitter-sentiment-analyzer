import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score, classification_report 
import pickle 
import os 

nltk.download('stopwords', quiet=True) 
nltk.download('wordnet', quiet=True) 

# ── Page config ────────────────────────────────────────────── 
st.set_page_config( 
    page_title="Twitter Sentiment Analyzer", 
    page_icon="🐦", 
    layout="wide" 
) 

# ── Styling ────────────────────────────────────────────────── 
st.markdown(""" 
<style> 
    .main { background-color: #0E1117; } 
    .stTextArea textarea { font-size: 16px; } 
    .positive { color: #00FF88; font-size: 24px; font-weight: bold; } 
    .negative { color: #FF4444; font-size: 24px; font-weight: bold; } 
    .neutral  { color: #FFAA00; font-size: 24px; font-weight: bold; } 
    .metric-card { 
        background: #1E2130; 
        border-radius: 10px; 
        padding: 20px; 
        text-align: center; 
        border: 1px solid #2E3250; 
    } 
</style> 
""", unsafe_allow_html=True) 

# ── Preprocessing ───────────────────────────────────────────── 
def preprocess(text): 
    text = text.lower() 
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs 
    text = re.sub(r'@\w+|#\w+', '', text)         # remove mentions/hashtags 
    text = re.sub(r'[^a-z\s]', '', text)           # remove special chars 
    tokens = text.split() 
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words] 
    return ' '.join(tokens) 

# ── Load or train model ─────────────────────────────────────── 
@st.cache_resource 
def load_model(): 
    # Sample training data — replace with your actual dataset 
    sample_data = { 
        'text': [ 
            "I love this product amazing great fantastic", 
            "This is wonderful best day ever happy", 
            "Great experience highly recommend excellent", 
            "Terrible awful horrible worst experience", 
            "I hate this so bad disgusting waste", 
            "Disappointing bad quality poor service", 
            "It's okay nothing special average mediocre", 
            "Not bad not great just fine neutral", 
            "Could be better could be worse alright" 
        ], 
        'label': ['Positive','Positive','Positive', 
                  'Negative','Negative','Negative', 
                  'Neutral','Neutral','Neutral'] 
    } 
    df = pd.DataFrame(sample_data) 
    df['clean'] = df['text'].apply(preprocess) 

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2)) 
    X = vectorizer.fit_transform(df['clean']) 
    y = df['label'] 

    nb_model  = MultinomialNB() 
    svm_model = LinearSVC() 
    nb_model.fit(X, y) 
    svm_model.fit(X, y) 

    return vectorizer, nb_model, svm_model 

vectorizer, nb_model, svm_model = load_model() 

# ── UI ──────────────────────────────────────────────────────── 
st.title("🐦 Twitter Sentiment Analyzer") 
st.markdown("**NLP Pipeline** — TF-IDF · Naive Bayes · SVM · Built by `https://hayredin.vercel.app` ") 
st.divider() 

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Tweet", "📊 Batch Analysis", "📈 Model Comparison"]) 

# ── Tab 1: Single tweet ─────────────────────────────────────── 
with tab1: 
    col1, col2 = st.columns([2, 1]) 
    with col1: 
        tweet = st.text_area("Enter a tweet to analyze:", 
            placeholder="Type any tweet here...", height=120) 
        model_choice = st.radio("Choose model:", 
            ["Naive Bayes", "SVM", "Both"], horizontal=True) 

        if st.button("Analyze Sentiment 🔍", use_container_width=True): 
            if tweet.strip(): 
                clean = preprocess(tweet) 
                vec   = vectorizer.transform([clean]) 

                if model_choice == "Naive Bayes": 
                    pred = nb_model.predict(vec)[0] 
                    proba = nb_model.predict_proba(vec)[0] 
                    labels = nb_model.classes_ 
                elif model_choice == "SVM": 
                    pred = svm_model.predict(vec)[0] 
                    proba = None 
                    labels = None 
                else: 
                    nb_pred  = nb_model.predict(vec)[0] 
                    svm_pred = svm_model.predict(vec)[0] 
                    pred = nb_pred 
                    proba = nb_model.predict_proba(vec)[0] 
                    labels = nb_model.classes_ 
                    st.info(f"**Naive Bayes:** {nb_pred} | **SVM:** {svm_pred}") 

                emoji = "✅" if pred=="Positive" else "❌" if pred=="Negative" else "⚪" 
                css   = pred.lower() 
                st.markdown(f'<p class="{css}">{emoji} {pred}</p>', 
                    unsafe_allow_html=True) 

                if proba is not None and labels is not None: 
                    fig, ax = plt.subplots(figsize=(6,3)) 
                    colors = ['#00FF88','#FF4444','#FFAA00'] 
                    bars = ax.barh(labels, proba, color=colors) 
                    ax.set_xlim(0,1) 
                    ax.set_facecolor('#1E2130') 
                    fig.patch.set_facecolor('#1E2130') 
                    ax.tick_params(colors='white') 
                    for bar, p in zip(bars, proba): 
                        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, 
                                f'{p:.1%}', va='center', color='white', fontsize=11) 
                    st.pyplot(fig) 
            else: 
                st.warning("Please enter a tweet first.") 

# ── Tab 2: Batch analysis ───────────────────────────────────── 
with tab2: 
    st.markdown("**Paste multiple tweets (one per line):**") 
    batch_input = st.text_area("", height=200, 
        placeholder="I love this!\nThis is terrible.\nIt's okay I guess.") 

    if st.button("Analyze All 📊", use_container_width=True): 
        if batch_input.strip(): 
            tweets = [t.strip() for t in batch_input.split('\n') if t.strip()] 
            results = [] 
            for t in tweets: 
                clean = preprocess(t) 
                vec   = vectorizer.transform([clean]) 
                nb    = nb_model.predict(vec)[0] 
                svm   = svm_model.predict(vec)[0] 
                results.append({'Tweet': t[:60]+'...' if len(t)>60 else t, 
                                 'Naive Bayes': nb, 'SVM': svm}) 

            df_results = pd.DataFrame(results) 
            st.dataframe(df_results, use_container_width=True) 

            # Pie chart 
            counts = df_results['Naive Bayes'].value_counts() 
            fig, ax = plt.subplots(figsize=(5,4)) 
            colors  = {'Positive':'#00FF88','Negative':'#FF4444','Neutral':'#FFAA00'} 
            ax.pie(counts.values, 
                   labels=counts.index, 
                   colors=[colors.get(l,'gray') for l in counts.index], 
                   autopct='%1.1f%%', 
                   textprops={'color':'white'}) 
            fig.patch.set_facecolor('#1E2130') 
            st.pyplot(fig) 

# ── Tab 3: Model comparison ─────────────────────────────────── 
with tab3: 
    st.markdown("### Naive Bayes vs SVM — Why both?") 
    col1, col2 = st.columns(2) 
    with col1: 
        st.markdown(""" 
        **Naive Bayes** 
        - Fast training, works well with small datasets 
        - Gives probability scores per class 
        - Best for: quick classification, probabilistic output 
        - Weakness: assumes feature independence 
        """) 
    with col2: 
        st.markdown(""" 
        **SVM (LinearSVC)** 
        - Finds optimal decision boundary 
        - Handles high-dimensional TF-IDF vectors well 
        - Best for: text classification accuracy 
        - Weakness: no probability output by default 
        """) 

    st.divider() 
    st.markdown("### Pipeline Architecture") 
    st.code(""" 
Raw Tweet 
    ↓ Lowercase, remove URLs/mentions/hashtags 
    ↓ Remove special characters 
    ↓ Tokenization 
    ↓ Stopword removal (NLTK) 
    ↓ Lemmatization (WordNetLemmatizer) 
    ↓ TF-IDF Vectorization (max 5000 features, bigrams) 
    ↓ Model prediction (Naive Bayes / SVM) 
    ↓ Sentiment label: Positive / Negative / Neutral 
    """, language="text") 

    st.markdown("*Built by `https://hayredin.vercel.app`  — github.com/HayreKhan750*")
