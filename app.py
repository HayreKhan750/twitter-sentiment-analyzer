import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from xquik_export import parse_xquik_export

# ---- Page Config ----
st.set_page_config(
    page_title="Twitter Sentiment Intel | AI Analytics",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom Premium CSS ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
    }
    
    /* Premium Header */
    .premium-header {
        background: linear-gradient(90deg, #1DA1F2 0%, #0077B5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 32px;
        text-align: center;
    }
    
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(90deg, #1DA1F2 0%, #0d8ddb 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 15px rgba(29, 161, 242, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass
    return set(stopwords.words('english'))

STOP_WORDS = setup_nltk()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text

@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_models()

# ---- Sidebar Branding ----
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h2 style='color: #1DA1F2; margin-bottom: 0;'>🐦 SENTIMENT INTEL</h2>
        <p style='color: #94a3b8; font-size: 12px;'>Developed by Hayredin</p>
    </div>
    <div style='text-align: center; padding-bottom: 20px;'>
        <a href='https://hayredin.vercel.app' target='_blank' style='text-decoration: none;'>
            <button style='background-color: #1DA1F2; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; width: 100%;'>
                🌐 Visit Portfolio
            </button>
        </a>
        <div style='margin-top: 15px;'>
            <a href='https://github.com/HayreKhan750' target='_blank' style='color: #94a3b8; text-decoration: none; font-size: 14px; margin-right: 15px;'>🐙 GitHub</a>
            <a href='https://linkedin.com' target='_blank' style='color: #94a3b8; text-decoration: none; font-size: 14px;'>💼 LinkedIn</a>
        </div>
    </div>
    <hr style='border: 0; border-top: 1px solid #334155; margin: 0 20px 20px 20px;'>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["⚡ Real-time Analysis", "📊 Analytics Hub", "📂 History Explorer"])

# ---- Initialization ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- Real-time Analysis ----
if page == "⚡ Real-time Analysis":
    st.markdown("<h1 class='premium-header'>Twitter Sentiment Intelligence</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Input Analysis")
        tweet_input = st.text_area("Analyze tweet sentiment in milliseconds:", placeholder="Enter a tweet or any text to decode its emotional signature...", height=150)
        
        analyze_btn = st.button("🚀 EXECUTE AI DECODER")
        
        if analyze_btn:
            if not tweet_input.strip():
                st.warning("⚠️ Please provide input for the engine.")
            elif model and vectorizer:
                with st.spinner("Decoding sentiment..."):
                    clean = clean_text(tweet_input)
                    vec = vectorizer.transform([clean])
                    pred = model.predict(vec)[0]
                    proba = model.predict_proba(vec)[0]
                    confidence = max(proba) * 100
                    
                    sentiment_label = "POSITIVE" if pred == 1 else "NEGATIVE"
                    sentiment_color = "#10b981" if pred == 1 else "#ef4444"
                    sentiment_emoji = "😊" if pred == 1 else "😞"
                    
                    st.session_state.history.append({
                        "text": tweet_input,
                        "sentiment": sentiment_label,
                        "confidence": confidence,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.markdown(f"""
                        <div style='background: #1e293b; padding: 30px; border-radius: 12px; border-left: 5px solid {sentiment_color};'>
                            <div style='color: #94a3b8; font-size: 14px; text-transform: uppercase;'>SENTIMENT RESULT</div>
                            <div style='color: {sentiment_color}; font-size: 48px; font-weight: 800;'>{sentiment_label} {sentiment_emoji}</div>
                            <div style='color: #f8fafc; font-size: 18px; margin-top: 10px;'>AI Confidence: <b>{confidence:.2f}%</b></div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if pred == 1: st.balloons()
            else:
                st.error("Engine failure: Models not detected.")

        st.markdown("### 📦 Xquik Export Batch")
        uploaded_export = st.file_uploader("Upload Xquik JSON, JSONL, or CSV export", type=["json", "jsonl", "csv"])
        if st.button("Analyze Xquik Export"):
            if uploaded_export is None:
                st.warning("Upload an export file first.")
            elif model and vectorizer:
                try:
                    export_texts = parse_xquik_export(uploaded_export.getvalue(), uploaded_export.name)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    for export_text in export_texts:
                        clean = clean_text(export_text)
                        vec = vectorizer.transform([clean])
                        pred = model.predict(vec)[0]
                        proba = model.predict_proba(vec)[0]
                        confidence = max(proba) * 100
                        st.session_state.history.append({
                            "text": export_text,
                            "sentiment": "POSITIVE" if pred == 1 else "NEGATIVE",
                            "confidence": confidence,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                    st.success(f"Analyzed {len(export_texts)} export rows.")
            else:
                st.error("Engine failure: Models not detected.")

    with col2:
        st.markdown("### 📈 Confidence Gauge")
        if st.session_state.history:
            last = st.session_state.history[-1]
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = last['confidence'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Analysis Confidence", 'font': {'size': 24, 'color': '#f8fafc'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#1DA1F2"},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 50], 'color': '#ef4444'},
                        {'range': [50, 80], 'color': '#f59e0b'},
                        {'range': [80, 100], 'color': '#10b981'}],
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#f8fafc", 'family': "Inter"}, height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting input for visualization...")

# ---- Analytics Hub ----
elif page == "📊 Analytics Hub":
    st.markdown("<h1 class='premium-header'>Sentiment Ecosystem</h1>", unsafe_allow_html=True)
    
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='metric-container'><div class='metric-label'>Total Analyzed</div><div class='metric-value'>{len(h_df)}</div></div>""", unsafe_allow_html=True)
        with c2:
            pos_count = len(h_df[h_df['sentiment'] == 'POSITIVE'])
            st.markdown(f"""<div class='metric-container'><div class='metric-label'>Positive Ratio</div><div class='metric-value'>{(pos_count/len(h_df)*100):.1f}%</div></div>""", unsafe_allow_html=True)
        with c3:
            avg_conf = h_df['confidence'].mean()
            st.markdown(f"""<div class='metric-container'><div class='metric-label'>Avg Confidence</div><div class='metric-value'>{avg_conf:.1f}%</div></div>""", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(h_df, names='sentiment', hole=.6, color='sentiment',
                            color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'},
                            title="Overall Sentiment Distribution")
            fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            fig_trend = px.line(h_df, x='timestamp', y='confidence', title="Analysis Confidence Trend",
                               line_shape='spline', markers=True)
            fig_trend.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("No analytics available. Run some analyses first!")

# ---- History Explorer ----
elif page == "📂 History Explorer":
    st.markdown("<h1 class='premium-header'>Intelligence Archives</h1>", unsafe_allow_html=True)
    
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        st.dataframe(h_df[['timestamp', 'sentiment', 'confidence', 'text']], use_container_width=True)
        
        st.download_button("📥 Export Intelligence Report (CSV)", 
                          data=h_df.to_csv(index=False).encode('utf-8'),
                          file_name=f"sentiment_report_{datetime.now().date()}.csv",
                          mime='text/csv', use_container_width=True)
    else:
        st.info("Archives are empty. Start analyzing to build your history.")

# ---- Footer ----
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #64748b; font-size: 14px;'>
        Built by <a href='https://hayredin.vercel.app' target='_blank' style='color: #1DA1F2; text-decoration: none;'>Hayredin</a> | 
        Powered by 🐦 Sentiment Intel AI | 
        <a href='https://github.com/HayreKhan750' target='_blank' style='color: #1DA1F2; text-decoration: none;'>GitHub</a> | 
        © {datetime.now().year} Enterprise Solutions
    </div>
""", unsafe_allow_html=True)
