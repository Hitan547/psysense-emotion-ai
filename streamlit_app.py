import streamlit as st
from inference import predict_emotions, plot_emotions, explain_emotion

st.set_page_config(page_title="PsySense Emotion AI", layout="centered")

# ---------------- UI HEADER ----------------

st.markdown(
    """
    <h1 style='text-align:center;'>🧠 PsySense Emotion AI Assistant</h1>
    <p style='text-align:center; font-size:18px;'>
    Understand your emotions • Get intelligent suggestions • Improve mental clarity
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- Conversation Memory ----------------

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ---------------- Advice Engine ----------------

def give_advice(emotion):

    advice_map = {

        "sadness": [
            "Take a short walk or get some fresh air.",
            "Talk to someone you trust.",
            "Write your thoughts in a journal."
        ],

        "fear": [
            "Try deep breathing for 2 minutes.",
            "Break your task into small steps.",
            "Avoid overthinking future outcomes."
        ],

        "anger": [
            "Pause before reacting.",
            "Do a quick physical activity.",
            "Reflect on what triggered the feeling."
        ],

        "joy": [
            "Use this positive energy to do productive work.",
            "Share your happiness with others.",
            "Practice gratitude."
        ],

        "love": [
            "Express appreciation to someone important.",
            "Strengthen your relationships.",
            "Do something meaningful together."
        ],

        "neutral": [
            "Set a small goal for today.",
            "Stay mindful of your routine.",
            "Plan something interesting."
        ]
    }

    return advice_map.get(emotion, [
        "Stay self-aware.",
        "Maintain emotional balance.",
        "Focus on positive habits."
    ])


# ---------------- Conversation Start ----------------

if not st.session_state.chat_started:

    st.subheader("💬 AI Assistant")
    st.write("How was your day today?")

    user_input = st.text_area("Share your feelings...")

    if st.button("Analyze My Emotions"):
        st.session_state.chat_started = True
        st.session_state.user_text = user_input
        st.rerun()


# ---------------- Emotion Analysis Section ----------------

else:

    text = st.session_state.user_text

    st.markdown(f"### 📝 You said: *{text}*")

    result = predict_emotions(text)
    st.session_state.last_result = result

    emotion = result["dominant_emotion"]["label"]
    confidence = result["dominant_emotion"]["confidence"]

    # Dominant Emotion Card
    st.markdown(
        f"""
        <div style="padding:15px;border-radius:10px;background:#1f3b4d;">
        <h3> Dominant Emotion: {emotion.capitalize()}</h3>
        <p>Confidence: {confidence*100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Explanation
    st.write("### 🧠 Emotion Insight")
    st.info(explain_emotion(emotion))

    # Multiple Emotions
    st.write("### 🔎 Other Detected Emotions")
    for e in result["active_emotions"]:
        if e["label"] != emotion:
            st.write(f"• {e['label'].capitalize()} — {e['confidence']*100:.2f}%")

    # Probability Breakdown
    st.write("### 📊 Emotion Probability Breakdown")
    for label, prob in result["top_emotions"]:
        if prob > 0.01:
            st.write(f"{label.capitalize()} — {prob*100:.2f}%")

    # Graph
    st.write("### 📈 Emotion Distribution Graph")
    fig = plot_emotions(result)
    st.pyplot(fig)

    # Advice Section
    st.write("### 💡 AI Suggestions for You")

    advice_list = give_advice(emotion)

    for tip in advice_list:
        st.success(tip)

    # Next Step Suggestion
    st.write("### 🚀 Suggested Next Step")

    st.markdown(
        """
        - Reflect on what caused this emotion  
        - Take a small positive action today  
        - Monitor how your mood changes over time  
        """
    )

    if st.button("Start New Conversation"):
        st.session_state.chat_started = False
        st.rerun()
