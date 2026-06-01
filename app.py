import streamlit as st
import numpy as np
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.datasets import imdb
import plotly.graph_objects as go
import plotly.express as px

# Ensure NLTK stopwords are downloaded quietly
@st.cache_resource
def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()
stop_words = set(stopwords.words('english'))

VOCAB_SIZE = 10000
MAX_LEN = 250

@st.cache_resource
def get_tokenizer():
    tokenizer_path = 'tokenizer.pickle'
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    
    with st.spinner("Building tokenizer (first-time setup). This takes a minute..."):
        # Load imdb to recreate the tokenizer
        (train_data, _), (_, _) = imdb.load_data(num_words=None)
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        def decode_review(text_integers):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_integers])
            
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^a-z\s]', '', text)
            words = text.split()
            words = [w for w in words if w not in stop_words]
            return ' '.join(words)
            
        cleaned_train_reviews = [clean_text(decode_review(review)) for review in train_data]
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<unk>')
        tokenizer.fit_on_texts(cleaned_train_reviews)
        
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return tokenizer

@st.cache_resource
def load_all_models():
    models = {}
    with st.spinner("Loading models..."):
        try:
            models['SimpleRNN'] = load_model('simple_rnn_model.h5')
            models['LSTM'] = load_model('lstm_model.h5')
            models['GRU'] = load_model('gru_model.h5')
        except Exception as e:
            st.error(f"Error loading models: {e}")
    return models

def preprocess_text(text, tokenizer):
    def clean_text(t):
        t = t.lower()
        t = re.sub(r'<.*?>', '', t)
        t = re.sub(r'[^a-z\s]', '', t)
        words = t.split()
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# UI Setup
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for a Premium Professional Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800;900&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Gradient Background for Hero */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        padding: 4rem 2rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    
    /* Subtle overlay pattern */
    .hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url('https://www.transparenttextures.com/patterns/cubes.png');
        opacity: 0.15;
        z-index: 0;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }

    .hero h1 {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 0 4px 15px rgba(0,0,0,0.2);
        letter-spacing: -2px;
        color: #ffffff;
    }
    
    .hero p {
        font-size: 1.4rem;
        font-weight: 500;
        opacity: 0.95;
        margin-top: 15px;
        text-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .tech-pill {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Floating Animation for Result Card */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .stTextArea textarea {
        background-color: #1a1c24 !important;
        border: 2px solid #2d313f !important;
        border-radius: 16px !important;
        padding: 20px !important;
        color: #fff !important;
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stTextArea textarea:focus {
        border-color: #23a6d5 !important;
        box-shadow: 0 0 20px rgba(35, 166, 213, 0.3) !important;
        transform: scale(1.01);
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff0844 0%, #ffb199 100%);
        color: #fff;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: 1px;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 2rem;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        box-shadow: 0 10px 20px rgba(255, 8, 68, 0.3);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 15px 30px rgba(255, 8, 68, 0.5);
    }
    
    .result-card {
        background: linear-gradient(145deg, #1e222d, #161820);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 24px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        animation: float 6s ease-in-out infinite;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(76, 175, 80, 0.05));
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.3);
        padding: 1.8rem;
        border-radius: 20px;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin: 1.5rem 0;
        text-transform: uppercase;
        letter-spacing: 3px;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.15);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15), rgba(244, 67, 54, 0.05));
        color: #f44336;
        border: 1px solid rgba(244, 67, 54, 0.3);
        padding: 1.8rem;
        border-radius: 20px;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin: 1.5rem 0;
        text-transform: uppercase;
        letter-spacing: 3px;
        box-shadow: 0 10px 30px rgba(244, 67, 54, 0.15);
    }
    .conf-text {
        font-size: 1.3rem;
        color: #aaa;
        font-weight: 500;
    }
    .model-stat-card {
        background: linear-gradient(145deg, #1e222d, #161820);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .model-stat-card:hover {
        transform: translateY(-10px);
        border-color: rgba(35, 166, 213, 0.5);
        box-shadow: 0 15px 40px rgba(35, 166, 213, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <h1>Movie Review Sentiment</h1>
        <p>State-of-the-art Natural Language Processing</p>
        <div class="tech-pill">✨ Powered by RNNs • LSTMs • GRUs</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize resources
tokenizer = get_tokenizer()
models = load_all_models()

# Layout
st.markdown("<h3 style='margin-bottom: 20px;'>🔍 Analyze Your Review</h3>", unsafe_allow_html=True)
col_input, col_output = st.columns([1.2, 1], gap="large")

with col_input:
    review_input = st.text_area(
        "Enter your movie review here...",
        height=280,
        placeholder="e.g. This movie was an absolute masterpiece! The cinematography was breathtaking, and the acting kept me on the edge of my seat the entire time...",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c3:
        analyze_btn = st.button("✨ Analyze Sentiment", use_container_width=True)

if analyze_btn and review_input.strip():
    if not models:
        st.error("Models failed to load. Please check the model files.")
    else:
        # Preprocess
        processed_input = preprocess_text(review_input, tokenizer)
        
        # Predictions
        preds = {}
        for name, model in models.items():
            pred = model.predict(processed_input)[0][0]
            preds[name] = float(pred)
            
        # Dynamically determine the best model (highest confidence)
        best_model = None
        max_confidence = -1
        
        for name, pred in preds.items():
            conf = pred if pred >= 0.5 else (1 - pred)
            if conf > max_confidence:
                max_confidence = conf
                best_model = name
                
        selected_pred = preds[best_model]
        sentiment = "Positive" if selected_pred >= 0.5 else "Negative"
        confidence = selected_pred if sentiment == "Positive" else (1 - selected_pred)
        
        with col_output:
            sentiment_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
            icon = "🌟" if sentiment == "Positive" else "💔"
            
            st.markdown(f"""
            <div class="result-card">
                <div style="color: #888; font-size: 1.1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">Highest Confidence: {best_model}</div>
                <div class="{sentiment_class}">{icon} {sentiment}</div>
                <div class="conf-text">Confidence Level: <span style="color: #fff; font-weight: 800; font-size: 1.5rem;">{confidence*100:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><hr style='border-color: #2d313f; margin: 40px 0;'><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 40px; font-weight: 800;'>📊 Comprehensive Model Comparison</h2>", unsafe_allow_html=True)
        
        # 3-Grid Cards
        comp_cols = st.columns(3)
        for i, (name, pred) in enumerate(preds.items()):
            is_pos = pred >= 0.5
            sent = "Positive" if is_pos else "Negative"
            conf = pred if is_pos else (1 - pred)
            color = "#23d5ab" if is_pos else "#ff0844"
            
            with comp_cols[i]:
                st.markdown(f"""
                <div class="model-stat-card">
                    <h3 style="margin:0; color: #fff; font-size: 1.6rem; font-weight: 800;">{name}</h3>
                    <p style="color: {color}; font-weight: 900; font-size: 1.4rem; margin: 15px 0;">{sent}</p>
                    <div style="width: 100%; background-color: #2d313f; border-radius: 10px; height: 12px; margin-top: 20px; overflow: hidden; box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);">
                        <div style="width: {conf*100}%; background-color: {color}; height: 100%; border-radius: 10px; transition: width 1.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);"></div>
                    </div>
                    <p style="color: #aaa; font-size: 1.1rem; margin-top: 20px; margin-bottom: 0; font-weight: 600;">Accuracy Match: <span style="color:#fff;">{conf*100:.1f}%</span></p>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Layout for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Donut chart
            fig_prob = go.Figure(data=[go.Pie(
                labels=['Positive Probability', 'Negative Probability'], 
                values=[selected_pred, 1 - selected_pred], 
                hole=.7,
                marker_colors=['#23d5ab', '#ff0844'],
                textinfo='label+percent',
                textfont=dict(size=15, color='white', family="Inter"),
                hoverinfo='label+percent'
            )])
            fig_prob.update_layout(
                title_text=f"Probability Distribution ({best_model})",
                title_x=0.5,
                height=380, 
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fff", family="Inter", size=14),
                showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with chart_col2:
            comp_df = {
                'Model': list(preds.keys()),
                'Positive Probability': [p * 100 for p in preds.values()],
                'Sentiment': ["Positive" if p >= 0.5 else "Negative" for p in preds.values()]
            }
            
            fig_comp = px.bar(
                comp_df, 
                x='Model', 
                y='Positive Probability', 
                color='Sentiment',
                color_discrete_map={'Positive': '#23d5ab', 'Negative': '#ff0844'},
                text='Positive Probability',
                range_y=[0, 100]
            )
            fig_comp.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                textfont=dict(size=14, family="Inter", color="white"),
                marker_line_color='rgba(0,0,0,0)',
                marker_line_width=0,
                width=0.45
            )
            fig_comp.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fff", family="Inter", size=14),
                title_text="Positive Probability Across All Models",
                title_x=0.5,
                xaxis_title="",
                yaxis_title="Probability (%)",
                showlegend=False
            )
            fig_comp.add_shape(
                type="line", line=dict(dash="dash", color="rgba(255,255,255,0.2)", width=2),
                x0=-0.5, x1=2.5, y0=50, y1=50
            )
            st.plotly_chart(fig_comp, use_container_width=True)

elif analyze_btn:
    st.warning("Please enter a review to analyze.")
