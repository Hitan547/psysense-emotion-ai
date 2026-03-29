import streamlit as st

# ── MUST be first Streamlit call — before any import that touches st ──
st.set_page_config(
    page_title="PsySense — Emotion AI",
    page_icon="🧠",
    layout="wide"
)

# ── Now safe to import inference ──────────────────────────────
from inference import (
    load_model, predict_emotions, plot_emotions,
    explain_emotion, get_emoji, give_advice
)

# ── Cache lives here in app, not in inference.py ─────────────
@st.cache_resource(show_spinner="Loading emotion model...")
def get_model():
    return load_model()

model, tokenizer, mlb, device = get_model()
label_names = list(mlb.classes_)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
# 🧠 PsySense — Emotion AI
### Understand what you're feeling, and what to do about it
""")
st.divider()

# ── Input ─────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    text    = st.text_area(
        "✍️ Share what's on your mind...",
        placeholder="Example: I feel proud but also nervous about tomorrow...",
        height=180
    )
    analyze = st.button("🔍 Analyze My Emotions", use_container_width=True)

with col2:
    st.info("""
### 💡 How it works
- Fine-tuned DistilBERT transformer
- Detects up to 28 distinct emotions
- Multi-label: multiple emotions at once
- Confidence scoring per emotion
- Personalized emotional advice
""")

st.divider()

# ── Results ───────────────────────────────────────────────────
if analyze:
    if not text or not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing your emotions..."):
            # Pass model components explicitly — no global state in inference.py
            result = predict_emotions(model, tokenizer, label_names, device, text)

        if "error" in result:
            st.error(result["error"])
        else:
            emotion    = result["dominant_emotion"]["label"]
            confidence = result["dominant_emotion"]["confidence"]
            emoji      = get_emoji(emotion)
            secondary  = [
                e for e in result["active_emotions"]
                if e["label"] != emotion
            ]

            # ── AI Insight ────────────────────────────────────
            st.markdown("## 🤖 AI Emotional Insight")

            if confidence < 0.35:
                st.warning(
                    f"💭 **Your emotions seem mixed or complex.**\n\n"
                    f"The strongest signal is **{emotion.capitalize()}** "
                    f"({confidence*100:.0f}% confidence), but it's not strongly dominant. "
                    f"This often happens when you're feeling several things at once."
                )
            elif secondary:
                blend = " and ".join(
                    f"**{e['label'].capitalize()}**" for e in secondary[:3]
                )
                st.success(
                    f"{emoji} **You seem to be feeling {emotion.capitalize()} "
                    f"alongside {blend}.**\n\n{explain_emotion(emotion)}"
                )
            else:
                st.success(
                    f"{emoji} **You seem to be feeling {emotion.capitalize()}.**\n\n"
                    f"{explain_emotion(emotion)}"
                )

            if secondary:
                parts = [f"**For your {em}:** {give_advice(em)}"
                         for em in ([emotion] + [e["label"] for e in secondary[:2]])]
                st.info("### 🌱 Suggested Next Steps\n\n" + "\n\n".join(parts))
            else:
                st.info(f"### 🌱 Suggested Next Step\n{give_advice(emotion)}")

            st.divider()

            # ── Dominant emotion ──────────────────────────────
            st.markdown("## 🎯 Dominant Emotion")
            if confidence < 0.35:
                st.caption("⚠️ Low confidence — emotions may be mixed")

            m1, m2, m3 = st.columns(3)
            m1.metric("Emotion",    emotion.capitalize())
            m2.metric("Confidence", f"{confidence*100:.1f}%")
            m3.metric("Emoji",      emoji)
            st.progress(min(confidence, 1.0))
            st.divider()

            # ── Secondary emotions ────────────────────────────
            st.markdown("## 🔁 Other Detected Emotions")
            if secondary:
                for e in secondary:
                    st.write(
                        f"{get_emoji(e['label'])} "
                        f"{e['label'].capitalize()} — "
                        f"{e['confidence']*100:.1f}%"
                    )
                    st.progress(min(e["confidence"], 1.0))
            else:
                st.caption("No secondary emotions detected above threshold.")

            st.divider()

            # ── Chart ─────────────────────────────────────────
            st.markdown("## 📊 Emotion Probability Distribution")
            st.pyplot(plot_emotions(result))
            st.divider()

            # ── Full breakdown ────────────────────────────────
            with st.expander("🔬 Full Probability Breakdown"):
                for label, prob in result["top_emotions"]:
                    if prob > 0.005:
                        c1, c2 = st.columns([2, 3])
                        c1.write(f"{get_emoji(label)} {label.capitalize()}")
                        c2.progress(min(prob, 1.0))

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
---
**PsySense AI** • Emotion Understanding + AI Suggestion System  
Built with DistilBERT Transformers & Streamlit
""")
