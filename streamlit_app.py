import streamlit as st
from inference import predict_emotions, plot_emotions, explain_emotion

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PsySense Emotion AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #f8fafc, #eef2ff);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #4f46e5;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🧠 PsySense Emotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Label Emotion Detection using DistilBERT</div>', unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
with st.container():
    st.markdown("### ✏️ Enter your text")
    text = st.text_area(
        "",
        height=150,
        placeholder="Example: I feel happy but also nervous about tomorrow..."
    )

    analyze = st.button("🚀 Analyze Emotion")

# ---------------- RESULT ----------------
if analyze:

    result = predict_emotions(text)

    if "error" in result:
        st.error(result["error"])

    else:
        emotion = result["dominant_emotion"]["label"]
        confidence = result["dominant_emotion"]["confidence"]

        st.markdown("---")

        # ⭐ Dominant Emotion Card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader(f"🎯 Dominant Emotion: {emotion.capitalize()}")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.info(explain_emotion(emotion))
        st.markdown('</div>', unsafe_allow_html=True)

        # ⭐ Other emotions
        st.markdown("### 🌈 Other Detected Emotions")

        cols = st.columns(3)

        idx = 0
        for e in result["active_emotions"]:
            if e["label"] != emotion:
                with cols[idx % 3]:
                    st.metric(
                        label=e["label"].capitalize(),
                        value=f"{e['confidence']*100:.1f}%"
                    )
                idx += 1

        # ⭐ Graph
        st.markdown("### 📊 Emotion Distribution")
        fig = plot_emotions(result)
        st.pyplot(fig)
