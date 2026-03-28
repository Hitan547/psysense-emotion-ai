import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — required for Streamlit
import matplotlib.pyplot as plt
import os
import streamlit as st

# ── Paths ────────────────────────────────────────────────────
HF_MODEL          = "Hitan2004/psysense-emotion-ai"
BASE_DIR          = os.path.dirname(__file__)
LOCAL_LABEL_ENC   = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)

# ── Model loading — cached so Streamlit never reloads it ─────
# BUG FIX: without this, the 400MB model reloads on every click
@st.cache_resource(show_spinner="Loading emotion model...")
def load_model():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = DistilBertForSequenceClassification.from_pretrained(HF_MODEL)
    tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL)
    model.to(device)
    model.eval()

    with open(LOCAL_LABEL_ENC, "rb") as f:
        mlb = pickle.load(f)

    print("✅ Model + tokenizer + label encoder loaded")
    return model, tokenizer, mlb, device

model, tokenizer, mlb, device = load_model()
label_names = mlb.classes_


# ── Prediction ───────────────────────────────────────────────
def predict_emotions(text, threshold=0.25, top_k=8):
    if not text or not text.strip():
        return {"error": "Empty input text"}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.sigmoid(logits)[0].cpu().numpy()
    sorted_idx = probs.argsort()[::-1]

    dominant_emotion = {
        "label":      label_names[sorted_idx[0]],
        "confidence": float(probs[sorted_idx[0]])
    }

    active_emotions = [
        {"label": label_names[i], "confidence": float(probs[i])}
        for i in sorted_idx
        if probs[i] >= threshold
    ]

    top_emotions = [
        (label_names[i], float(probs[i]))
        for i in sorted_idx[:top_k]
    ]

    return {
        "dominant_emotion": dominant_emotion,
        "active_emotions":  active_emotions,
        "top_emotions":     top_emotions
    }


# ── Explanations — all 28 emotions covered ───────────────────
# BUG FIX: old version had only 6, everything else showed "Emotion detected."
EXPLANATIONS = {
    "admiration":    "You're expressing genuine respect or appreciation for someone.",
    "amusement":     "Something struck you as funny or entertaining.",
    "anger":         "You're experiencing strong frustration or rage.",
    "annoyance":     "Something is irritating or mildly frustrating you.",
    "approval":      "You're feeling agreement or positive acknowledgment.",
    "caring":        "You're showing concern, kindness, or emotional support.",
    "confusion":     "You're feeling uncertain or finding something hard to understand.",
    "curiosity":     "You're interested in learning or discovering something new.",
    "desire":        "You're strongly wanting or longing for something.",
    "disappointment":"Your expectations weren't met — that's a valid feeling.",
    "disapproval":   "You're expressing disagreement or criticism about something.",
    "disgust":       "Something is triggering strong dislike or revulsion in you.",
    "embarrassment": "You're feeling awkward or ashamed about something.",
    "excitement":    "You're filled with enthusiasm or positive anticipation.",
    "fear":          "You're experiencing anxiety, worry, or dread.",
    "gratitude":     "You're feeling thankful and appreciative.",
    "grief":         "You're experiencing deep sorrow or emotional loss.",
    "joy":           "You're feeling happiness and positive energy.",
    "love":          "You're experiencing deep affection or emotional attachment.",
    "nervousness":   "You're feeling tension or nervous anticipation.",
    "optimism":      "You're hopeful and expecting positive outcomes.",
    "pride":         "You're feeling accomplished or satisfied with yourself.",
    "realization":   "You just had a moment of sudden understanding or insight.",
    "relief":        "A burden has lifted — you're feeling comfort after stress.",
    "remorse":       "You're feeling guilt or regret about something.",
    "sadness":       "You're experiencing emotional pain or sorrow.",
    "surprise":      "Something unexpected caught you off guard.",
    "neutral":       "Your text doesn't carry strong emotional charge."
}

def explain_emotion(label):
    return EXPLANATIONS.get(label, "An emotion was detected in your text.")


# ── Emoji map ────────────────────────────────────────────────
EMOJI_MAP = {
    "admiration": "🤩", "amusement": "😂", "anger": "😠",
    "annoyance": "😤",  "approval": "👍",  "caring": "🤗",
    "confusion": "😕",  "curiosity": "🤔", "desire": "😍",
    "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
    "embarrassment": "😳", "excitement": "🎉", "fear": "😨",
    "gratitude": "🙏",  "grief": "😢",     "joy": "😊",
    "love": "❤️",       "nervousness": "😰", "optimism": "🌟",
    "pride": "🏆",      "realization": "💡", "relief": "😮‍💨",
    "remorse": "😔",    "sadness": "😢",    "surprise": "😲",
    "neutral": "😐"
}

def get_emoji(label):
    return EMOJI_MAP.get(label, "💭")


# ── Visualization ────────────────────────────────────────────
# BUG FIX: was returning plt (module) — now returns fig object
# This prevents Streamlit double-render warnings and memory leaks
def plot_emotions(result, min_prob=0.01):
    labels = []
    scores = []

    for label, prob in result["top_emotions"]:
        if prob >= min_prob:
            labels.append(label.capitalize())
            scores.append(prob)

    colors = [
        "#2ecc71" if s >= 0.5 else "#f39c12" if s >= 0.25 else "#e74c3c"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=0.6)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title("Emotion Probability Distribution", fontsize=13, fontweight="bold")
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.8, label="Threshold (0.25)")

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val*100:.0f}%",
            ha="center", va="bottom", fontsize=9
        )

    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig            # ← return fig, NOT plt
