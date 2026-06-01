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
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-top: 5px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .sentiment-positive {
        color: #4caf50;
        font-weight: 800;
        font-size: 1.8rem;
    }
    .sentiment-negative {
        color: #f44336;
        font-weight: 800;
        font-size: 1.8rem;
    }
    div[data-baseweb="textarea"] > div {
        background-color: #2b2b2b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Movie Review Sentiment Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Based Sentiment Classification</div>', unsafe_allow_html=True)

# Initialize resources
tokenizer = get_tokenizer()
models = load_all_models()

# Layout
col_input, col_output = st.columns([1.5, 1], gap="large")

with col_input:
    st.markdown("### Input Area")
    review_input = st.text_area(
        "Enter your movie review here...",
        height=200,
        placeholder="e.g. This movie was an absolute masterpiece! The acting was incredible and the plot kept me on the edge of my seat..."
    )
    
    c1, c2 = st.columns(2)
    with c1:
        selected_model = st.selectbox("Select Model for Primary Analysis", ["GRU", "LSTM", "SimpleRNN"])
    with c2:
        st.write("") # spacing to align with selectbox
        st.write("")
        analyze_btn = st.button("Analyze Review", type="primary", use_container_width=True)

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
            
        selected_pred = preds[selected_model]
        sentiment = "Positive" if selected_pred >= 0.5 else "Negative"
        confidence = selected_pred if sentiment == "Positive" else (1 - selected_pred)
        
        with col_output:
            st.markdown("### Output Area")
            sentiment_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-bottom: 5px; color: #fff;">Sentiment</h4>
                <div class="{sentiment_class}">{sentiment}</div>
                <p style="margin-top:10px; color:#aaa; font-size: 1.1rem;">Confidence: <strong style="color: #fff;">{confidence*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Visualization Area")
            # Donut chart for Positive vs Negative probability
            fig_prob = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative'], 
                values=[selected_pred, 1 - selected_pred], 
                hole=.6,
                marker_colors=['#4caf50', '#f44336'],
                textinfo='label+percent'
            )])
            fig_prob.update_layout(
                title_text=f"{selected_model} Confidence Chart",
                title_x=0.5,
                height=300, 
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fff")
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### Model Comparison Dashboard")
        
        # Create a bar chart comparing all three models
        comp_df = {
            'Model': list(preds.keys()),
            'Positive Probability (%)': [p * 100 for p in preds.values()],
            'Sentiment': ["Positive" if p >= 0.5 else "Negative" for p in preds.values()]
        }
        
        fig_comp = px.bar(
            comp_df, 
            x='Model', 
            y='Positive Probability (%)', 
            color='Sentiment',
            color_discrete_map={'Positive': '#4caf50', 'Negative': '#f44336'},
            text='Positive Probability (%)',
            range_y=[0, 100]
        )
        fig_comp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_comp.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff"),
            title_text="Prediction Comparison Across All Models",
            title_x=0.5
        )
        fig_comp.add_shape(
            type="line", line=dict(dash="dash", color="white", width=2),
            x0=-0.5, x1=2.5, y0=50, y1=50
        )
        st.plotly_chart(fig_comp, use_container_width=True)
elif analyze_btn:
    st.warning("Please enter a review to analyze.")
