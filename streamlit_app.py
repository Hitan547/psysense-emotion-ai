import streamlit as st
from inference import predict_emotions, plot_emotions, explain_emotion

st.set_page_config(page_title="PsySense Emotion AI", layout="centered")

st.title("🧠 PsySense Emotion AI")
st.write("Enter text to detect emotions")

text = st.text_area("Enter your sentence")

if st.button("Analyze Emotion"):

    result = predict_emotions(text)

    if "error" in result:
        st.error(result["error"])

    else:
        emotion = result["dominant_emotion"]["label"]
        confidence = result["dominant_emotion"]["confidence"]

        st.subheader(f"Dominant Emotion: {emotion.capitalize()}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        st.write("Explanation:")
        st.info(explain_emotion(emotion))

        # ⭐ SHOW OTHER EMOTIONS
        st.write("Other Detected Emotions:")
        for e in result["active_emotions"]:
            if e["label"] != emotion:
                st.write(f"{e['label']} — {e['confidence']*100:.2f}%")

        # ⭐ PROBABILITY TABLE
        st.write("Emotion Probability Breakdown:")
        for label, prob in result["top_emotions"]:
            if prob > 0.01:
                st.write(f"{label} — {prob*100:.2f}%")

        # ⭐ GRAPH
        fig = plot_emotions(result)
        st.pyplot(fig)
