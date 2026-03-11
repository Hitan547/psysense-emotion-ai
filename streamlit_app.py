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

        st.subheader(f"Dominant Emotion: {emotion}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        st.write("Explanation:")
        st.info(explain_emotion(emotion))

        st.write("Emotion Distribution:")
        plot_emotions(result)
