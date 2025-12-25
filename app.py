import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import base64
from io import BytesIO
from datetime import datetime
import os

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- VOICE OPTIONS ---
voice_options = {
    "Scribe v1": "pNInz6obpgDQGcFmaJgB",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Josh": "TxGEqnHWrfWFTfGW9XjX"
}

# --- LANGCHAIN LLM SETUP ---
llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
prompt_template = PromptTemplate(
    input_variables=["topic", "genre", "tone"],
    template="Write a {tone} {genre} story about: {topic}"
)
story_chain = LLMChain(llm=llm, prompt=prompt_template)

# --- FUNCTIONS ---
def generate_story(topic, genre, tone):
    result = story_chain.invoke({"topic": topic, "genre": genre, "tone": tone})
    return result['text'] if isinstance(result, dict) and 'text' in result else result

def generate_image(prompt):
    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"prompt": prompt, "n": 1, "size": "512x512"}
    )
    if response.status_code == 200:
        return response.json()['data'][0]['url']
    return None

def generate_audio(story, voice_id):
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        },
        json={"text": story, "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}}
    )
    if response.status_code == 200:
        return BytesIO(response.content)
    return None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Multimodal Content Generator", layout="centered")
st.title("🚀 Multimodal Content Generator")
st.write("Enter a prompt and customize your experience!")

user_input = st.text_input("Enter your creative prompt:", placeholder="A dragon who learns to dance")
genre = st.selectbox("Choose a genre", ["Fantasy", "Sci-Fi", "Horror", "Comedy"])
tone = st.selectbox("Choose the tone", ["Uplifting", "Suspenseful", "Melancholy"])
image_style = st.selectbox("Choose image style", ["Photorealistic", "Oil Painting", "Anime"])
selected_voice = st.selectbox("Choose a voice for narration", list(voice_options.keys()))

if st.button("Generate") and user_input:
    with st.spinner("Generating story..."):
        story = generate_story(user_input, genre, tone)
    st.subheader("📖 Story")
    st.write(story)
    st.download_button("Download Story", data=str(story).encode(), file_name="story.txt")

    with st.spinner("Generating image..."):
        image_prompt = f"A {image_style.lower()} illustration of {user_input}"
        image_url = generate_image(image_prompt)
        if image_url:
            st.subheader("🖼️ Image")
            st.image(image_url)
        else:
            st.error("Failed to generate image.")

    with st.spinner("Generating audio..."):
        audio_data = generate_audio(story, voice_options[selected_voice])
        if audio_data:
            st.subheader("🔊 Audio Narration")
            st.audio(audio_data, format="audio/mp3")
            st.download_button("Download Audio", data=audio_data, file_name="narration.mp3")
        else:
            st.error("Failed to generate audio.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, OpenAI, and ElevenLabs")
