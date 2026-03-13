import streamlit as st
from inference import predict_emotions, plot_emotions, explain_emotion

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PsySense Emotion AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- AI ADVICE ENGINE ----------------
def give_advice(emotion):

    advice_map = {

        "sadness": "It seems like you're feeling low. Consider talking to a friend or taking a short walk. Small steps can help improve mood.",

        "fear": "Try deep breathing or grounding techniques. Writing down what worries you may help reduce anxiety.",

        "anger": "Take a pause before reacting. Physical activity or journaling can help release emotional tension.",

        "nervousness": "Preparation and structured thinking can reduce nervousness. Try focusing on what you can control.",

        "joy": "That’s wonderful! Try to share this positive energy with someone or use it to work on something meaningful.",

        "love": "Connection is powerful. Express gratitude and strengthen your relationships.",

        "neutral": "Your emotional state seems balanced. This can be a great time to focus on productivity or learning.",

        "disappointment": "Reflect on expectations vs reality. Use this as learning for future growth.",

        "grief": "Give yourself time to process emotions. Talking to someone supportive can help healing."
    }

    return advice_map.get(
        emotion,
        "Try mindfulness, rest, or discussing your feelings with someone you trust."
    )

# ---------------- HERO SECTION ----------------
st.markdown("""
# 🧠 PsySense Emotion AI  
### Understand Human Emotions from Text  

Detect **dominant and secondary emotions** using a fine-tuned Transformer model.
""")

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns([2,1])

with col1:
    text = st.text_area(
        "✍️ How was your day today?",
        placeholder="Example: I feel proud but also nervous about tomorrow...",
        height=180
    )

    analyze = st.button("🔍 Analyze My Emotions", use_container_width=True)

with col2:
    st.info("""
### 💡 How it works
- Multi-label emotion detection  
- Transformer based NLP model  
- Confidence scoring  
- Emotion visualization  
- AI emotional suggestions  
""")

st.divider()

# ---------------- RESULT SECTION ----------------
if analyze:

    result = predict_emotions(text)

    if "error" in result:
        st.error(result["error"])

    else:
        emotion = result["dominant_emotion"]["label"]
        confidence = result["dominant_emotion"]["confidence"]

        # ---------------- AI RESPONSE ----------------
        st.markdown("## 🤖 AI Emotional Insight")

        st.success(
            f"I understand you may be feeling **{emotion}**. "
            f"{explain_emotion(emotion)}"
        )

        st.info(
            "### 🌱 Suggested Next Step\n"
            + give_advice(emotion)
        )

        st.divider()

        # ---- DOMINANT EMOTION CARD ----
        st.markdown("## 🎯 Dominant Emotion")

        card_col1, card_col2 = st.columns([1,3])

        with card_col1:
            st.metric(
                label="Emotion",
                value=emotion.capitalize()
            )

        with card_col2:
            st.progress(confidence)
            st.write(f"Confidence Score: **{confidence*100:.2f}%**")

        st.divider()

        # ---- SECONDARY EMOTIONS ----
        st.markdown("##  Other Detected Emotions")

        emotion_tags = []
        for e in result["active_emotions"]:
            if e["label"] != emotion:
                emotion_tags.append(
                    f"**{e['label'].capitalize()}** ({e['confidence']*100:.1f}%)"
                )

        if emotion_tags:
            st.write(" | ".join(emotion_tags))
        else:
            st.write("No strong secondary emotions detected.")

        st.divider()

        # ---- PROBABILITY BREAKDOWN ----
        st.markdown("## 📊 Emotion Probability Distribution")

        for label, prob in result["top_emotions"]:
            if prob > 0.01:
                st.write(f"{label.capitalize():15} — {prob*100:.2f}%")

        fig = plot_emotions(result)
        st.pyplot(fig)

        st.divider()

# ---------------- FOOTER ----------------
st.markdown("""
---
**PsySense AI • Emotion Understanding + AI Suggestion System**  
Built  using Transformers & Streamlit  
""")
