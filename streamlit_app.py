import streamlit as st
from inference import predict_emotions, plot_emotions, explain_emotion

st.set_page_config(
    page_title="PsySense Emotion AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(120deg,#0f172a,#020617);
    color: #e5e7eb;
}

/* Center container */
.main-container {
    max-width: 900px;
    margin: auto;
    padding-top: 40px;
}

/* Title */
.title {
    font-size: 46px;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 40px;
}

/* Text box */
textarea {
    background-color: #020617 !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    padding: 15px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    background: linear-gradient(90deg,#6366f1,#22d3ee);
    border: none;
    color: white;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(99,102,241,0.4);
}

/* Result Card */
.result-card {
    background: #020617;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #1e293b;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 PsySense Emotion AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Understand emotions from text using Transformer Intelligence</div>", unsafe_allow_html=True)
text = st.text_area(
    "How was your day?",
    placeholder="Example: I feel proud but also nervous about tomorrow..."
)

if st.button("🔎 Analyze Emotion"):

    result = predict_emotions(text)

    if "error" in result:
        st.error(result["error"])

    else:

        emotion = result["dominant_emotion"]["label"]
        confidence = result["dominant_emotion"]["confidence"]

        # ---------- RESULT CARD ----------
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.markdown(f"### Dominant Emotion → **{emotion.capitalize()}**")

        st.progress(float(confidence))

        st.write(f"Confidence Score: **{confidence*100:.1f}%**")

        st.write("### 🧠 Explanation")
        st.info(explain_emotion(emotion))

        # ---------- OTHER EMOTIONS ----------
        st.write("### 🌈 Other Detected Emotions")

        cols = st.columns(4)

        i = 0
        for e in result["active_emotions"]:
            if e["label"] != emotion:
                cols[i % 4].markdown(
                    f"""
                    <div style="
                        background:#020617;
                        padding:10px;
                        border-radius:10px;
                        border:1px solid #1e293b;
                        text-align:center;
                        ">
                        <b>{e['label'].capitalize()}</b><br>
                        {e['confidence']*100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                i += 1

        # ---------- GRAPH ----------
        st.write("### 📊 Emotion Distribution")

        fig = plot_emotions(result)
        st.pyplot(fig)

        # ---------- AI ADVICE ----------
        st.write("### 🤖 AI Suggestion")

        if emotion in ["sadness","grief","remorse","disappointment"]:
            st.success("Try talking to someone you trust. Small positive activities can help shift emotional state.")

        elif emotion in ["fear","nervousness"]:
            st.success("Take deep breaths and focus on what you can control. Preparation reduces anxiety.")

        elif emotion in ["anger","annoyance"]:
            st.success("Pause before reacting. Physical movement or short breaks help release emotional tension.")

        elif emotion in ["joy","love","gratitude","pride"]:
            st.success("Great emotional state 🙂 Try journaling or sharing this positivity with others.")

        else:
            st.success("Stay mindful of your emotions. Awareness is the first step toward emotional intelligence.")

        st.markdown("</div>", unsafe_allow_html=True)
