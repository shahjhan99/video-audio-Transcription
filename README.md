# 🎧 Speaker Diarization and Transcription Web App
---

## 📄 **Detailed Description (README.md)**

````markdown
# 🎧 Speaker Diarization and Transcription Web App

This Gradio-based web application allows users to upload video or audio files and get:
- A full **transcription** with **speaker labels**
- A **visual diarization plot**
- A **summary** of the conversation
- **Suggested interview questions** based on the dialogue content

---

## 🚀 Features
- 🔊 **Speaker Diarization** using `pyannote-audio@2.1`
- 🗣️ **Speech Transcription** using OpenAI’s `whisper` (base model)
- 🤖 **LLM Analysis** using `Groq` API with `llama3-8b-8192` model
- 📈 **Diarization Timeline Plot**
- 📋 **Language Detection** for unknown languages
- 📁 Downloadable JSON diarization output

---

## 🧠 Models and Tools Used
| Feature            | Tool/Model                   |
|--------------------|------------------------------|
| Transcription      | `OpenAI Whisper (base)`      |
| Speaker Diarization| `pyannote/speaker-diarization@2.1` |
| Summarization & Qs | `Groq API - llama3-8b-8192`  |
| Frontend UI        | `Gradio`                     |
| Video Processing   | `moviepy`                    |
| Plotting           | `matplotlib`                 |
| Language Detection | `langdetect`                 |

---

## ⚙️ Requirements

Install all dependencies via pip:

```bash
pip install -r requirements.txt
````

### You'll also need:

* A Hugging Face API Token with **access to gated models**
* A Groq API Key

---

## 🔐 Authentication Notes

To run the app successfully, make sure:

* You have [enabled access to gated repos](https://huggingface.co/settings/tokens) on Hugging Face for `pyannote/speaker-diarization`
* Your Groq token has permissions to access `llama3-8b-8192`

---

## 🧪 How to Run

```bash
python app.py
```

Or run inside a Jupyter/Colab Notebook or host using `gradio` interface.

---

## 👨‍💻 Developer

**Muhammad Shahjhan Gondal**
📧 [shahjhangondal99@gmail.com](mailto:shahjhangondal99@gmail.com)

````

---

## 📦 `requirements.txt`

```txt
gradio==4.30.0
whisper
pyannote.audio==2.1.1
moviepy
langdetect
matplotlib
tqdm
groq
huggingface_hub
````

> 📌 Note: Some packages like `whisper` may require `ffmpeg`. Install it system-wide using:

```bash
sudo apt install ffmpeg
```

---

