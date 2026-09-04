import streamlit as st
import torch
import numpy as np
import shap
import os
import time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from google import genai

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------------------------------------------------------------------------
# Cached resource loading (model, tokenizer, SHAP explainer, Gemini client)
# ---------------------------------------------------------------------------

MODEL_PATH = "./distilbert_liar_final"
MAX_LENGTH = 64
GEMINI_MODEL = "gemini-flash-lite-latest"  # update if your key requires a different model name


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_explainer(_tokenizer, _model, _device):
    def predict_proba(texts, batch_size=8):
        all_probs = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = _tokenizer(
                batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(_device)
            with torch.no_grad():
                logits = _model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    masker = shap.maskers.Text(_tokenizer)
    explainer = shap.Explainer(predict_proba, masker)
    return predict_proba, explainer


@st.cache_resource
def load_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY", "API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def get_top_shap_tokens(explainer, text, class_idx, top_k=5):
    shap_values = explainer([text])
    tokens = shap_values.data[0]
    values = shap_values.values[0, :, class_idx]
    pairs = list(zip(tokens, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_k]


def build_explanation_prompt(text, predicted_label, confidence, top_tokens):
    fake_words = [t.strip() for t, v in top_tokens if v > 0]
    real_words = [t.strip() for t, v in top_tokens if v < 0]

    return f"""The model classified this statement as [{predicted_label.upper()}] with {confidence:.0%} confidence.

Statement: "{text}"

The most influential words pushing toward FAKE were: {fake_words if fake_words else "none"}.
The most influential words pushing toward REAL were: {real_words if real_words else "none"}.

Write a 2-sentence explanation for a non-technical reader, referencing the specific influential words above. Do not introduce reasoning that isn't grounded in these words."""


def generate_explanation(client, prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return f"(Explanation unavailable: {e})"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📰 Fake News Detector")
st.caption("Enter a news headline or statement to check its credibility, with an explanation of why.")

with st.spinner("Loading model..."):
    tokenizer, model, device = load_model()
    predict_proba, explainer = load_explainer(tokenizer, model, device)
    gemini_client = load_gemini_client()

if gemini_client is None:
    st.warning(
        "GEMINI_API_KEY environment variable not set — plain-English explanations will be unavailable. "
        "SHAP word importances will still be shown."
    )

user_text = st.text_area(
    "Enter a headline or statement",
    placeholder="e.g. Says the new bill will double property taxes for homeowners.",
    height=100,
)

analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked and user_text.strip():
    with st.spinner("Analyzing..."):
        probs = predict_proba([user_text])[0]
        pred_class = int(probs.argmax())
        confidence = float(probs.max())
        predicted_label = "Fake" if pred_class == 1 else "Real"

        top_tokens = get_top_shap_tokens(explainer, user_text, class_idx=pred_class, top_k=5)

    # --- Prediction result ---
    st.subheader("Result")
    col1, col2 = st.columns(2)
    with col1:
        if predicted_label == "Fake":
            st.error(f"**Prediction: {predicted_label}**")
        else:
            st.success(f"**Prediction: {predicted_label}**")
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")

    # --- SHAP word importances ---
    st.subheader("Most Influential Words")
    for token, val in top_tokens:
        direction = "toward FAKE" if val > 0 else "toward REAL"
        color = "🔴" if val > 0 else "🟢"
        st.write(f"{color} **{token.strip()}** — {direction} ({val:+.3f})")

    # --- LLM explanation ---
    st.subheader("Explanation")
    if gemini_client is not None:
        prompt = build_explanation_prompt(user_text, predicted_label, confidence, top_tokens)
        explanation = generate_explanation(gemini_client, prompt)
        st.info(explanation)
    else:
        st.write("Set the `GEMINI_API_KEY` environment variable to enable plain-English explanations.")

elif analyze_clicked:
    st.warning("Please enter some text to analyze.")

st.divider()
st.caption(
    "Model: DistilBERT fine-tuned on the LIAR dataset. "
    "Explanations generated using SHAP attributions passed to Google's Gemini API."
)
