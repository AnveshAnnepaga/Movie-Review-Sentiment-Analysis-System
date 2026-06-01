# Movie Review Sentiment Analysis System 🎬

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

**🚀 [Live Demo Available Here](https://movie-review-sentiment-analysis-system-aj9xfanupux76rhxhpappfc.streamlit.app/)**

A professional, deep learning-based web application for analyzing the sentiment of movie reviews. The system uses a trained Recurrent Neural Network architecture (with options to compare SimpleRNN, LSTM, and GRU) to classify whether a movie review is positive or negative.

## Features ✨
- **Beautiful User Interface**: A modern, dark-themed UI built with Streamlit and styled with custom CSS.
- **Deep Learning Models**: Powered by Keras and TensorFlow, using standard SimpleRNN, LSTM, and GRU networks.
- **Real-Time Prediction**: Instantly parses user input and returns sentiment along with confidence probabilities.
- **Model Comparison Dashboard**: Visualizes and compares the prediction accuracy and probability outputs of all three models on the same review using interactive Plotly charts.
- **Auto-Configuring Tokenizer**: Re-compiles the exact sequence padding and tokenization based on the IMDB standard dataset on first run.

## Installation 🚀

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AnveshAnnepaga/Movie-Review-Sentiment-Analysis-System.git
cd Movie-Review-Sentiment-Analysis-System
pip install -r requirements.txt
```

## Running the App 🖥️

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

## Deployment on Streamlit Community Cloud ☁️
This repository is pre-configured for deployment on Streamlit Cloud.
1. Connect your GitHub repository to Streamlit Community Cloud.
2. Ensure the `Main file path` is set to `app.py`.
3. The `.python-version` file ensures that Python 3.11 is used to maintain compatibility with TensorFlow.

## Author
**Anvesh** - [GitHub](https://github.com/AnveshAnnepaga)