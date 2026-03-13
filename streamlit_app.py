import streamlit as st
from inference import predict_emotions, plot_emotions, explain_emotion
import matplotlib.pyplot as plt

st.set_page_config(page_title="PsySense Emotion AI", layout="wide")

# ---------- THEME ----------

st.markdown("""

<style>
.main {
background: linear-gradient(120deg,#020617,#020617);
color:white;
}
.chat-user {
background:#1e293b;
padding:12px;
border-radius:12px;
margin:8px 0;
}
.chat-ai {
background:#0f172a;
padding:12px;
border-radius:12px;
margin:8px 0;
border-left:4px solid #38bdf8;
}
.big-title {
font-size:42px;
font-weight:700;
}
.subtitle {
color:#94a3b8;
font-size:18px;
}
</style>

""", unsafe_allow_html=True)

# ---------- SESSION MEMORY ----------

# ---------- SESSION STATE ----------

if "history" not in st.session_state:
    st.session_state.history = []

if "mood_scores" not in st.session_state:
    st.session_state.mood_scores = []

# ---------- HEADER ----------

st.markdown("<div class='big-title'>🧠 PsySense Emotion Coach</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI that understands how you feel</div>", unsafe_allow_html=True)

st.divider()

# ---------- INPUT ----------

user_text = st.text_input("How was your day?")

if st.button("Analyze Emotion"):

```
result = predict_emotions(user_text)

if "error" in result:
    st.error(result["error"])

else:

    emotion = result["dominant_emotion"]["label"]
    confidence = result["dominant_emotion"]["confidence"]

    # Save history
    st.session_state.history.append((user_text, emotion, confidence))
    st.session_state.mood_scores.append(confidence)
```

# ---------- CHAT HISTORY ----------

for text, emo, conf in st.session_state.history:

```
st.markdown(f"<div class='chat-user'>🙂 {text}</div>", unsafe_allow_html=True)

st.markdown(
    f"<div class='chat-ai'>🤖 Detected Emotion: <b>{emo.capitalize()}</b> ({conf*100:.1f}%)</div>",
    unsafe_allow_html=True
)

st.info(explain_emotion(emo))

# Advice engine
if emo in ["sadness","grief","remorse","disappointment"]:
    st.success("💡 Try talking to someone you trust or go for a short walk.")

elif emo in ["fear","nervousness"]:
    st.success("💡 Focus on preparation and breathing exercises.")

elif emo in ["anger","annoyance"]:
    st.success("💡 Pause before reacting. Physical activity helps release tension.")

elif emo in ["joy","love","gratitude","pride"]:
    st.success("💡 Great state 🙂 Use this energy for productive work.")

else:
    st.success("💡 Stay mindful. Emotional awareness builds intelligence.")

st.divider()
```

# ---------- MOOD TREND ----------

if len(st.session_state.mood_scores) > 1:

```
st.subheader("📈 Mood Confidence Trend")

fig2 = plt.figure(figsize=(6,3))
plt.plot(st.session_state.mood_scores, marker="o")
plt.title("Emotion Confidence Over Time")
plt.xlabel("Conversation Step")
plt.ylabel("Confidence")
plt.tight_layout()

st.pyplot(fig2)
```
