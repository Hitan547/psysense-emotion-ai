import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)

# ------------------ MODEL LOADING ------------------

HF_MODEL = "Hitan2004/psysense-emotion-ai"

BASE_DIR = os.path.dirname(__file__)
LOCAL_LABEL_ENCODER = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(HF_MODEL)
model.to(device)
model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL)

# ⭐ load local file
with open(LOCAL_LABEL_ENCODER, "rb") as f:
    mlb = pickle.load(f)

label_names = mlb.classes_

print("✅ Model + tokenizer + label encoder loaded")


# ------------------ PREDICTION FUNCTION ------------------

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

    probs = torch.sigmoid(logits)[0].cpu().numpy()
    sorted_idx = probs.argsort()[::-1]

    dominant_emotion = {
        "label": label_names[sorted_idx[0]],
        "confidence": float(probs[sorted_idx[0]])
    }

    active_emotions = [
        {
            "label": label_names[i],
            "confidence": float(probs[i])
        }
        for i in sorted_idx
        if probs[i] >= threshold
    ]

    top_emotions = [
        (label_names[i], float(probs[i]))
        for i in sorted_idx[:top_k]
    ]

    return {
        "dominant_emotion": dominant_emotion,
        "active_emotions": active_emotions,
        "top_emotions": top_emotions
    }


# ------------------ EXPLANATION ------------------

def explain_emotion(label):

    explanations = {
        "joy": "The text expresses happiness or positive feelings.",
        "sadness": "The text expresses sadness or emotional pain.",
        "fear": "The text shows anxiety or worry.",
        "anger": "The text reflects frustration or anger.",
        "love": "The text conveys affection or care.",
        "neutral": "The text does not express strong emotion."
    }

    return explanations.get(label, "Emotion detected.")


# ------------------ EMOJI ------------------

emotion_emoji = {
    "joy": "😊",
    "sadness": "😔",
    "fear": "😨",
    "anger": "😠",
    "love": "❤️",
    "pride": "🏆",
    "neutral": "😐"
}


# ------------------ DISPLAY ------------------

def display_result(result):

    emotion = result["dominant_emotion"]["label"]
    confidence = result["dominant_emotion"]["confidence"]

    emoji = emotion_emoji.get(emotion, "")

    print(f"\nDetected Emotion: {emotion.capitalize()} {emoji}")
    print(f"Confidence: {confidence*100:.1f}%")

    print("\nExplanation:")
    print(explain_emotion(emotion))

    if result["active_emotions"]:
        print("\nOther Detected Emotions:")
        for e in result["active_emotions"]:
            if e["label"] != emotion:
                print(f"• {e['label']} — {e['confidence']*100:.1f}%")

    print("\nEmotion Probability Breakdown:")
    for label, prob in result["top_emotions"]:
        if prob > 0.01:
            print(f"{label:15} {prob*100:.1f}%")


# ------------------ VISUALIZATION ------------------

def plot_emotions(result):

    labels = []
    scores = []

    for label, prob in result["top_emotions"]:
        if prob > 0.01:
            labels.append(label)
            scores.append(prob)

    plt.figure(figsize=(8,4))
    plt.bar(labels, scores)
    plt.xticks(rotation=45)
    plt.title("Emotion Probability Distribution")
    plt.ylabel("Confidence")
    plt.tight_layout()
    return plt
