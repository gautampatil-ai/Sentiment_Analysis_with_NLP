"""
Sentiment Analysis App
----------------------
A Streamlit interface for a TF-IDF + Logistic Regression sentiment
classifier (positive / negative).

Run locally:
    streamlit run app.py
"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            padding-top: 1.5rem;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 0.75rem;
            text-align: center;
            margin-top: 1rem;
        }
        .result-positive {
            background-color: #e6f4ea;
            border: 1px solid #34a853;
        }
        .result-negative {
            background-color: #fce8e6;
            border: 1px solid #ea4335;
        }
        .result-label {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .result-sub {
            font-size: 0.95rem;
            color: #555;
        }
        .stTextArea textarea {
            font-size: 1rem;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model loading (cached so it only runs once per session)
# --------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"
VECTORIZER_PATH = Path(__file__).parent / "vectorizer.pkl"


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_sentiment(text: str, model, vectorizer):
    """Return (label, probability_dict) for a single piece of text."""
    features = vectorizer.transform([text])
    label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    proba_dict = dict(zip(model.classes_, proba))
    return label, proba_dict


try:
    model, vectorizer = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("💬 About")
    st.write(
        "This app uses a **TF-IDF vectorizer** paired with a "
        "**Logistic Regression** classifier to predict whether a piece "
        "of text expresses **positive** or **negative** sentiment."
    )
    st.divider()
    st.subheader("How it works")
    st.markdown(
        """
        1. Enter or paste some text
        2. Click **Analyze Sentiment**
        3. View the predicted label and confidence score
        """
    )
    st.divider()
    st.caption("Built with Streamlit · scikit-learn")

# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
st.title("💬 Sentiment Analyzer")
st.write(
    "Analyze the sentiment of any piece of text — reviews, feedback, "
    "tweets, or comments — using a machine learning model."
)

if not artifacts_loaded:
    st.error(
        "Model files not found. Make sure `model.pkl` and "
        "`vectorizer.pkl` are in the same folder as `app.py`."
    )
    st.stop()

st.markdown("#### Enter your text")

example_texts = {
    "-- Select an example --": "",
    "Positive example": "This product exceeded all my expectations, I absolutely love it!",
    "Negative example": "This was a complete waste of money, terrible experience.",
}

selected_example = st.selectbox("Try an example (optional)", list(example_texts.keys()))

user_input = st.text_area(
    "Text to analyze",
    value=example_texts[selected_example],
    height=150,
    placeholder="Type or paste your text here...",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)
with col2:
    char_count = len(user_input)
    st.caption(f"{char_count} characters")

# --------------------------------------------------------------------------
# Prediction & results
# --------------------------------------------------------------------------
if analyze_clicked:
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            label, proba_dict = predict_sentiment(user_input, model, vectorizer)

        confidence = proba_dict[label]
        is_positive = str(label).lower() == "positive"
        card_class = "result-positive" if is_positive else "result-negative"
        emoji = "😊" if is_positive else "😞"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label">{emoji} {str(label).capitalize()}</div>
                <div class="result-sub">Confidence: {confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Class probabilities")

        proba_df = pd.DataFrame(
            {"Sentiment": list(proba_dict.keys()), "Probability": list(proba_dict.values())}
        ).sort_values("Probability", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=proba_df["Probability"],
                y=proba_df["Sentiment"].str.capitalize(),
                orientation="h",
                marker_color=[
                    "#34a853" if s.lower() == "positive" else "#ea4335"
                    for s in proba_df["Sentiment"]
                ],
                text=[f"{p:.1%}" for p in proba_df["Probability"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1], tickformat=".0%", title=None),
            yaxis=dict(title=None),
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ Predictions are generated by a statistical model and may not "
    "always be accurate. Use results as guidance, not ground truth."
)
