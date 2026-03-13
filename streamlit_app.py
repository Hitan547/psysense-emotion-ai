import streamlit as st
import time
from inference import predict_emotions, plot_emotions, explain_emotion

st.set_page_config(page_title="PsySense Emotion AI", layout="wide")

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "mood_scores" not in st.session_state:
    st.session_state.mood_scores = []

# ---------- UI HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>🧠 PsySense Emotion AI</h1>
<h4 style='text-align:center;color:gray;'>Understand human emotions from text</h4>
""", unsafe_allow_html=True)

st.divider()

# ---------- CHAT STYLE INPUT ----------
st.subheader("💬 How was your day?")

user_text = st.text_area(
    "Share your feelings",
    placeholder="Example: I feel very stressed and tired today..."
)

analyze = st.button("🔎 Analyze Emotion", use_container_width=True)

# ---------- ANALYSIS ----------
if analyze:

    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:

        with st.spinner("Analyzing emotions..."):
            time.sleep(1)
            result = predict_emotions(user_text)

        emotion = result["dominant_emotion"]["label"]
        confidence = result["dominant_emotion"]["confidence"]

        # Save history
        st.session_state.history.append((user_text, emotion))
        st.session_state.mood_scores.append(confidence)

        # ---------- RESULT CARD ----------
        st.markdown("### 🎯 Dominant Emotion")
        col1, col2 = st.columns([1,2])

        with col1:
            st.metric(label="Emotion", value=emotion.capitalize())
            st.metric(label="Confidence", value=f"{confidence*100:.1f}%")

        with col2:
            st.info(explain_emotion(emotion))

        st.divider()

        # ---------- OTHER EMOTIONS ----------
        st.markdown("### 🌈 Other Detected Emotions")

        emotions = result["active_emotions"]

        if len(emotions) > 1:
            for e in emotions:
                if e["label"] != emotion:
                    st.progress(float(e["confidence"]))
                    st.write(f"{e['label']} — {e['confidence']*100:.1f}%")
        else:
            st.write("No strong secondary emotions")

        st.divider()

        # ---------- GRAPH ----------
        st.markdown("### 📊 Emotion Distribution")
        fig = plot_emotions(result)
        st.pyplot(fig)

        st.divider()

        # ---------- AI ADVICE ----------
        st.markdown("### 🤖 AI Suggestion")

        advice_map = {
            "sadness": "Try talking to a friend or going for a short walk.",
            "fear": "Take deep breaths. Focus on what you can control.",
            "anger": "Pause before reacting. Write your thoughts down.",
            "joy": "Great! Capture this moment and keep building on it.",
            "love": "Express gratitude to the person you care about.",
            "nervousness": "Prepare small steps. Confidence grows gradually.",
            "disappointment": "Reflect on what you learned and move forward."
        }

        advice = advice_map.get(
            emotion,
            "Stay mindful. Emotions are temporary signals."
        )

        st.success(advice)

# ---------- HISTORY ----------
if len(st.session_state.history) > 0:

    st.divider()
    st.markdown("### 🕘 Conversation History")

    for h in reversed(st.session_state.history[-5:]):
        st.write(f"**You:** {h[0]}")
        st.write(f"**Emotion:** {h[1]}")
        st.write("---")
