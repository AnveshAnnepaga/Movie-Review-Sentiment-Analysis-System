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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .hero {
        background: linear-gradient(105deg, #0f2027, #203a43, #2c5364);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: -webkit-linear-gradient(45deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.8;
        margin-top: 10px;
    }
    .stTextArea textarea {
        background-color: #1a1c24 !important;
        border: 1px solid #2d313f !important;
        border-radius: 12px !important;
        padding: 15px !important;
        color: #fff !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #00C9FF !important;
        box-shadow: 0 0 10px rgba(0, 201, 255, 0.2) !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #111;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 201, 255, 0.4);
    }
    .result-card {
        background: #1a1c24;
        border: 1px solid #2d313f;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .sentiment-positive {
        background: rgba(76, 175, 80, 0.1);
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin: 1.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .sentiment-negative {
        background: rgba(244, 67, 54, 0.1);
        color: #f44336;
        border: 1px solid rgba(244, 67, 54, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin: 1.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .conf-text {
        font-size: 1.2rem;
        color: #aaa;
        font-weight: 500;
    }
    .model-stat-card {
        background: #1a1c24;
        border: 1px solid #2d313f;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .model-stat-card:hover {
        transform: translateY(-5px);
        border-color: #00C9FF;
        box-shadow: 0 8px 20px rgba(0, 201, 255, 0.15);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Movie Review Sentiment Analysis</h1>
    <p>Advanced Deep Learning Classification Powered by RNNs, LSTMs, and GRUs</p>
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
        height=250,
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
                <div style="color: #888; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Best Performing Model: {best_model}</div>
                <div class="{sentiment_class}">{icon} {sentiment}</div>
                <div class="conf-text">Model Confidence: <span style="color: #fff; font-weight: 700; font-size: 1.4rem;">{confidence*100:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><hr style='border-color: #2d313f;'><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin-bottom: 30px;'>📊 Comprehensive Model Comparison</h3>", unsafe_allow_html=True)
        
        # 3-Grid Cards
        comp_cols = st.columns(3)
        for i, (name, pred) in enumerate(preds.items()):
            is_pos = pred >= 0.5
            sent = "Positive" if is_pos else "Negative"
            conf = pred if is_pos else (1 - pred)
            color = "#00C9FF" if is_pos else "#FF416C"
            
            with comp_cols[i]:
                st.markdown(f"""
                <div class="model-stat-card">
                    <h3 style="margin:0; color: #fff; font-size: 1.5rem;">{name}</h3>
                    <p style="color: {color}; font-weight: 800; font-size: 1.3rem; margin: 10px 0;">{sent}</p>
                    <div style="width: 100%; background-color: #2d313f; border-radius: 10px; height: 10px; margin-top: 15px; overflow: hidden;">
                        <div style="width: {conf*100}%; background-color: {color}; height: 100%; border-radius: 10px; transition: width 1s ease-in-out;"></div>
                    </div>
                    <p style="color: #aaa; font-size: 1rem; margin-top: 15px; margin-bottom: 0; font-weight: 500;">Confidence: <span style="color:#fff;">{conf*100:.1f}%</span></p>
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
                marker_colors=['#00C9FF', '#FF416C'],
                textinfo='label+percent',
                textfont=dict(size=14, color='white'),
                hoverinfo='label+percent'
            )])
            fig_prob.update_layout(
                title_text=f"Probability Distribution ({best_model})",
                title_x=0.5,
                height=350, 
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fff", family="Inter"),
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
                color_discrete_map={'Positive': '#00C9FF', 'Negative': '#FF416C'},
                text='Positive Probability',
                range_y=[0, 100]
            )
            fig_comp.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                marker_line_color='rgba(0,0,0,0)',
                marker_line_width=0,
                width=0.4
            )
            fig_comp.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fff", family="Inter"),
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
