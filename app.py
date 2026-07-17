"""
Review Sentiment Analyzer
--------------------------
An animated Streamlit app for sentiment classification using a
TF-IDF Vectorizer + Logistic Regression model.

Run locally with:
    streamlit run app.py
"""

import pickle
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_lottie import st_lottie

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "model.pkl"
VECTORIZER_PATH = Path(__file__).parent / "vectorizer.pkl"
LOTTIE_URL = "https://assets10.lottiefiles.com/packages/lf20_v7v9mjat.json"

SENTIMENT_COLORS = {"positive": "#21c45d", "negative": "#ef4444"}
SENTIMENT_EMOJI = {"positive": "😊", "negative": "😠"}


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_data(show_spinner=False, ttl=3600)
def load_lottieurl(url: str):
    """Fetch a Lottie animation JSON, failing gracefully if unavailable."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None


def predict_sentiment(text: str, model, vectorizer):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
    return prediction, proba


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("✨ Sentiment Analyzer")
    st.markdown(
        """
        Analyze the tone of customer reviews using a
        **TF-IDF + Logistic Regression** model.

        **How to use:**
        1. Paste or type a review.
        2. Click **Analyze Sentiment**.
        3. View the predicted label and confidence chart.
        """
    )
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Plotly")


# --------------------------------------------------------------------------
# Load model + animation
# --------------------------------------------------------------------------
try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `model.pkl` and `vectorizer.pkl` "
        "are in the same directory as `app.py`."
    )
    st.stop()

lottie_anim = load_lottieurl(LOTTIE_URL)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
if lottie_anim:
    st_lottie(lottie_anim, height=200, key="coding")

st.title("✨ Review Sentiment Analyzer")
st.markdown("Analyze the tone of your customer reviews with high precision.")

# --------------------------------------------------------------------------
# Input section
# --------------------------------------------------------------------------
st.subheader("Enter your review below:")
user_input = st.text_area(
    "Write something...",
    placeholder="Type here...",
    height=150,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

# --------------------------------------------------------------------------
# Prediction and results
# --------------------------------------------------------------------------
if analyze_clicked:
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.6)  # brief visual effect
            prediction, proba = predict_sentiment(user_input, model, vectorizer)

        st.divider()
        sentiment = str(prediction).lower()
        emoji = SENTIMENT_EMOJI.get(sentiment, "")

        if sentiment == "positive":
            st.balloons()
            st.success(f"### The Sentiment is: **{sentiment.upper()}** {emoji}")
        elif sentiment == "negative":
            st.error(f"### The Sentiment is: **{sentiment.upper()}** {emoji}")
        else:
            st.info(f"### The Sentiment is: **{sentiment.upper()}**")

        # Confidence chart
        if proba is not None:
            st.subheader("Confidence Score")
            df_prob = pd.DataFrame([proba], columns=model.classes_)
            df_long = df_prob.T.reset_index()
            df_long.columns = ["Sentiment", "Probability"]

            fig = px.bar(
                df_long,
                x="Probability",
                y="Sentiment",
                orientation="h",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                text_auto=".1%",
                range_x=[0, 1],
            )
            fig.update_layout(
                showlegend=False,
                xaxis_tickformat=".0%",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()
