import streamlit as st
from inference import (
    predict_emotions, plot_emotions,
    explain_emotion, get_emoji
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PsySense — Emotion AI",
    page_icon="🧠",
    layout="wide"
)

# ── Advice engine — all 28 emotions covered ──────────────────
ADVICE = {
    "sadness":       "You don't have to carry this alone. Reach out to someone you trust, or try journaling your feelings.",
    "grief":         "Grief takes time. Be gentle with yourself and allow the process to unfold at its own pace.",
    "fear":          "Try box breathing: inhale 4s, hold 4s, exhale 4s. Writing down fears often reduces their power.",
    "nervousness":   "Focus on what you can control. Preparation and grounding techniques can quiet nervous energy.",
    "anger":         "Pause before reacting. Physical movement or journaling can help release built-up tension.",
    "annoyance":     "A short break or change of environment can reset your perspective on small frustrations.",
    "disappointment":"Reflect on what you learned. Every unmet expectation is a chance to refine your approach.",
    "disapproval":   "It's okay to disagree. Consider whether it's worth expressing — and how to do so constructively.",
    "disgust":       "Distance yourself from what triggered this. Your reaction is valid — trust your instincts.",
    "remorse":       "Acknowledging a mistake takes courage. Consider what you can do to make amends and move forward.",
    "embarrassment": "Everyone has awkward moments. They feel bigger to you than they look to others.",
    "confusion":     "Break the problem into smaller pieces. Asking for clarification is always a sign of strength.",
    "joy":           "Wonderful! Share this energy with someone, or use it to work on something meaningful to you.",
    "excitement":    "Channel this energy into action — you're in a great state to start something new.",
    "love":          "Connection is one of life's greatest gifts. Express gratitude and nurture your relationships.",
    "gratitude":     "Gratitude is a superpower. Consider telling the person you're grateful for how you feel.",
    "admiration":    "Let them know — expressing admiration can strengthen bonds and inspire others.",
    "pride":         "You've earned this. Take a moment to acknowledge your effort before moving to the next goal.",
    "optimism":      "This is a great mindset. Use it to tackle something you've been putting off.",
    "caring":        "Your empathy is a strength. Make sure you're also caring for yourself, not just others.",
    "curiosity":     "Follow that thread! Curiosity is the engine of learning — explore it.",
    "desire":        "Clarify what you want, then think about one small step you can take toward it today.",
    "amusement":     "Laughter is medicine. Spread it — share what made you smile.",
    "surprise":      "Take a moment to process before reacting. Surprises can be good or bad — give it space.",
    "relief":        "Take a deep breath and enjoy this lighter feeling. You've made it through something hard.",
    "realization":   "Capture this insight before it fades — write it down or share it with someone.",
    "neutral":       "A calm state is a great time for focused work, reflection, or learning something new.",
}

def give_advice(emotion):
    return ADVICE.get(emotion, "Try mindfulness, rest, or talking to someone you trust.")


# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
# 🧠 PsySense — Emotion AI
### Understand what you're feeling, and what to do about it
""")
st.divider()


# ── Input ────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area(
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


# ── Results ──────────────────────────────────────────────────
if analyze:
    if not text or not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        # Spinner gives user feedback while model runs
        with st.spinner("Analyzing your emotions..."):
            result = predict_emotions(text)

        if "error" in result:
            st.error(result["error"])

        else:
            emotion    = result["dominant_emotion"]["label"]
            confidence = result["dominant_emotion"]["confidence"]
            emoji      = get_emoji(emotion)

            # ── AI Insight ───────────────────────────────────────────────
st.markdown("## 🤖 AI Emotional Insight")

emotion    = result["dominant_emotion"]["label"]
confidence = result["dominant_emotion"]["confidence"]
emoji      = get_emoji(emotion)

# Collect all active emotions excluding dominant
secondary = [
    e for e in result["active_emotions"]
    if e["label"] != emotion
]

# ── Case 1: Low confidence dominant — model is uncertain ─────
if confidence < 0.35:
    st.warning(
        f"💭 **Your emotions seem mixed or complex.**\n\n"
        f"The strongest signal detected is **{emotion.capitalize()}** "
        f"({confidence*100:.0f}% confidence), but it's not strongly dominant. "
        f"This often happens when you're feeling several things at once."
    )

# ── Case 2: High confidence + secondary emotions ─────────────
elif secondary:
    secondary_labels = [
        f"**{e['label'].capitalize()}**" for e in secondary[:3]
    ]
    blend_str = " and ".join(secondary_labels)
    st.success(
        f"{emoji} **You seem to be feeling {emotion.capitalize()} "
        f"alongside {blend_str}.**\n\n"
        f"{explain_emotion(emotion)}"
    )

# ── Case 3: Single clear emotion ─────────────────────────────
else:
    st.success(
        f"{emoji} **You seem to be feeling {emotion.capitalize()}.**\n\n"
        f"{explain_emotion(emotion)}"
    )

# ── Blended advice when multiple emotions present ─────────────
if secondary:
    all_emotions = [emotion] + [e["label"] for e in secondary[:2]]
    advice_parts = []

    for em in all_emotions:
        advice = give_advice(em)
        advice_parts.append(f"**For your {em}:** {advice}")

    st.info(
        "### 🌱 Suggested Next Steps\n" +
        "\n\n".join(advice_parts)
    )
else:
    st.info(
        f"### 🌱 Suggested Next Step\n{give_advice(emotion)}"
    )

st.divider()

# ── Dominant emotion ──────────────────────────────────────────
st.markdown("## 🎯 Dominant Emotion")

# Warn user if confidence is low
if confidence < 0.35:
    st.caption("⚠️ Low confidence — emotions may be mixed or ambiguous")

m1, m2, m3 = st.columns(3)
m1.metric("Emotion",    emotion.capitalize())
m2.metric("Confidence", f"{confidence*100:.1f}%")
m3.metric("Emoji",      emoji)
st.progress(min(confidence, 1.0))
st.divider()

# ── Secondary emotions ────────────────────────────────────────
st.markdown("## 🔁 Other Detected Emotions")

if secondary:
    for e in secondary:
        label_text = (
            f"{get_emoji(e['label'])}  "
            f"{e['label'].capitalize()} — "
            f"{e['confidence']*100:.1f}%"
        )
        st.write(label_text)
        st.progress(min(e["confidence"], 1.0))
else:
    st.caption("No secondary emotions detected above threshold.")
```
