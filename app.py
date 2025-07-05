import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
import os
import uuid

model = tf.keras.models.load_model("model/best_model22.keras")
label_classes = np.load("model/label_classes2.npy")

FIXED_LENGTH = 16000  
SAMPLE_RATE = 16000

emotion_reactions = {
    "happy": [
        "🌟 Embrace the good vibes!",
        "💬 Share your joy with someone you love.",
        "🎶 Dance it out with your favorite song!"
    ],
    "sad": [
        "🌧️ Let yourself feel — it's okay.",
        "🛏️ Take a break, you deserve rest.",
        "🧘 Breathe. This too shall pass."
    ],
    "angry": [
        "🔥 Channel it into something positive.",
        "🚶 Go for a short walk or stretch it out.",
        "🎨 Try journaling or drawing your emotions."
    ],
    "fear": [
        "🔒 You’re safe now. Deep breaths.",
        "🤝 Talk to someone — you’re not alone.",
        "🧠 Acknowledge your thoughts, gently."
    ],
    "neutral": [
        "🧘 You're balanced — maintain it.",
        "🌅 Reflect on your thoughts calmly.",
        "☕ Enjoy a quiet moment for yourself."
    ],
    "disgust": [
        "😬 That didn’t sit right — it’s valid to feel this way.",
        "🚿 Distance from negativity, refresh your space.",
        "🌸 Focus on what brings beauty or peace to your senses."
    ]
}

def extract_logmel_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=1.0)
    y = librosa.util.fix_length(y, size=FIXED_LENGTH)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel)
    return log_mel[np.newaxis, ..., np.newaxis]  

st.set_page_config(page_title="🎙️ Emotion Detector", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>🎧 Speech Emotion Recognition App</h1>
    <p style='text-align: center;'>Turn your voice into emotion insights using AI & Deep Learning</p>
""", unsafe_allow_html=True)

st.markdown("### 🎵 Upload a WAV file to begin:")
uploaded_file = st.file_uploader("Upload your speech sample", type=["wav"], label_visibility="collapsed")

if uploaded_file:
    try:
        unique_id = str(uuid.uuid4())
        audio_dir = "audio"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"{unique_id}.wav")

        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(audio_path, format="audio/wav")

        features = extract_logmel_features(audio_path)
        prediction = model.predict(features)
        emotion = label_classes[np.argmax(prediction)]

        st.markdown("### 🧠 Detected Emotion: **<span style='color:#3c82f6;'>{}</span>**".format(emotion.upper()), unsafe_allow_html=True)

        with st.expander("💡 Click for personalized suggestions", expanded=True):
            for idea in emotion_reactions.get(emotion.lower(), ["💪 Stay strong and take care of yourself."]):
                st.markdown(f"- {idea}")

        st.markdown("### 📊 Emotion Probabilities")
        prob_df = pd.DataFrame({
            "Emotion": label_classes,
            "Probability": prediction[0]
        }).sort_values(by="Probability", ascending=False)

        st.bar_chart(data=prob_df.set_index("Emotion"))

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
