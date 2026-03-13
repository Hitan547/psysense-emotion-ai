import streamlit as st
from inference import predict_emotions, plot_emotions, explain_emotion

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PsySense Emotion AI",
    page_icon="🧠",
    layout="wide"
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
        "✍️ Enter your text",
        placeholder="Example: I feel proud but also nervous about tomorrow...",
        height=180
    )

    analyze = st.button("🔍 Analyze Emotion", use_container_width=True)

with col2:
    st.info("""
### 💡 How it works
- Multi-label emotion detection  
- Transformer based model  
- Confidence scoring  
- Emotion visualization  
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

        st.success(explain_emotion(emotion))

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
**PsySense AI • Multi-Label Emotion Intelligence System**  
Built using Transformers & Streamlit  
""")
