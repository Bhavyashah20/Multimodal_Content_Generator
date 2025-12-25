
# Multimodal Content Generator

A web application built with Streamlit, LangChain, OpenAI, and ElevenLabs that generates a short story, a matching image, and an audio narration based on a user's creative prompt.

## 🌟 Features
- **AI-generated story** using OpenAI's language models
- **Image creation** via OpenAI's image generation API (DALL·E)
- **Voice narration** using ElevenLabs text-to-speech
- **Interactive UI** for selecting genre, tone, image style, and narration voice

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- ElevenLabs API key

### Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-link-here>
   cd multimodal-content-generator
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your environment variables:
   - Create a `.env` file in the project root and add:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ELEVENLABS_API_KEY=your_elevenlabs_api_key
     ```

### Run the App
```bash
streamlit run app.py
```

## 📄 Usage
1. Enter a creative prompt (e.g., "A dragon who learns to dance").
2. Choose your preferred genre, tone, image style, and narration voice.
3. Click **Generate** to create a story, an illustration, and a voiceover.

## 🧰 Tech Stack
- [Streamlit](https://streamlit.io)
- [LangChain](https://www.langchain.com)
- [OpenAI GPT & DALL·E APIs](https://platform.openai.com)
- [ElevenLabs Text-to-Speech](https://www.elevenlabs.io)

## 📎 Project Link
```
<https://chatgpt.com/c/6812406e-35d8-8000-b99a-3c86c426ce84>
```

## 💡 License
MIT License

## 🙏 Acknowledgments
- OpenAI for the GPT and image generation APIs
- ElevenLabs for their speech synthesis API
- Streamlit for easy and elegant web UI

---
Built with ❤️ using AI technologies

=======
# Multimodal_Content_Generator

