import gradio as gr
import whisper
import os
from pyannote.audio import Pipeline
from datetime import timedelta
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from groq import Groq
from langdetect import detect, LangDetectException



#------------------------------ Initialize models and clients


HUGGINGFACE_TOKEN = "hf_**********************"                                   #  Replace with your Actual HUGGINGFACE_TOKEN-api-key
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=HUGGINGFACE_TOKEN)
model = whisper.load_model("base")
groq_client = Groq(api_key="gsk_*******************************0U6D")             #  Replace with your Actual Groq Api key for loading model

def extract_audio(video_path, output_path="audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path)
    return output_path

def transcribe_video_or_audio(file_path):
    if file_path.endswith((".mp3", ".wav")):
        audio_path = file_path
        video_path = None
    else:
        audio_path = extract_audio(file_path)
        video_path = file_path

    diarization = pipeline(audio_path)
    asr_result = model.transcribe(audio_path)
    asr_segments = asr_result["segments"]

    final_transcript = []
    speaker_map = {}
    next_speaker_id = 1

    for i, segment in enumerate(tqdm(asr_segments, desc="Mapping Speakers")):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        speaker = "Unknown"
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if not (end < turn.start or start > turn.end):
                if spk not in speaker_map:
                    speaker_map[spk] = f"SPEAKER_{next_speaker_id:02d}"
                    next_speaker_id += 1
                speaker = speaker_map[spk]
                break

# ------------------------------Language Detection

        unknown_lang_tag = ""
        if text:
            try:
                lang = detect(text)
                if lang != "en" and (i == len(asr_segments) - 1 or detect(asr_segments[i + 1]["text"]) == "en"):
                    unknown_lang_tag = f" [UNKNOWN LANGUAGE: {lang}]"
            except LangDetectException:
                unknown_lang_tag = " [UNKNOWN LANGUAGE: unknown]"
        else:
            text = "[No speech detected]"
            unknown_lang_tag = ""

        start_time = str(timedelta(seconds=int(start)))
        end_time = str(timedelta(seconds=int(end)))
        line = f"[{speaker}] ({start_time} - {end_time}): {text}{unknown_lang_tag}"
        final_transcript.append(line)

#------------------------------ Insert silence or noise gap

        if i > 0:
            prev_end = asr_segments[i - 1]["end"]
            if start - prev_end > 1.0:
                silence_start = str(timedelta(seconds=int(prev_end)))
                silence_end = str(timedelta(seconds=int(start)))
                silence_line = f"[NO SPEECH] ({silence_start} - {silence_end}): [Silence or background noise detected]"
                final_transcript.append(silence_line)

    with open("final_transcript.txt", "w", encoding="utf-8") as f:
        for line in final_transcript:
            f.write(line + "\n")

    diarization_data = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_data.append({
            "start_time": round(turn.start, 2),
            "end_time": round(turn.end, 2),
            "speaker": speaker_map.get(speaker, speaker)
        })
    with open("diarization_output.json", "w", encoding="utf-8") as f:
        json.dump(diarization_data, f, indent=4)

    start_times = [turn.start for turn, _, _ in diarization.itertracks(yield_label=True)]
    end_times = [turn.end for turn, _, _ in diarization.itertracks(yield_label=True)]
    speakers = [speaker_map.get(spk, spk) for _, _, spk in diarization.itertracks(yield_label=True)]

    plt.figure(figsize=(10, 5))
    for i in range(len(start_times)):
        plt.plot([start_times[i], end_times[i]], [i, i], label=speakers[i], marker='o')

    plt.xlabel('Time (s)')
    plt.ylabel('Speaker')
    plt.title('Speaker Diarization')
    plt.yticks(range(len(start_times)), speakers)
    plt.grid(True)
    plt.savefig("diarization_plot.png")
    plt.close()

    return "\n".join(final_transcript), "diarization_output.json", "diarization_plot.png", video_path

def generate_summary(transcript_text):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an expert assistant. Analyze the following transcript."},
            {"role": "user", "content": f"Summarize this speaker-labeled transcript:\n\n{transcript_text}"}
        ]
    )
    summary = response.choices[0].message.content
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    return summary

def generate_questions(transcript_text):
    prompt = f"""Here is a transcript of an interview with labeled speakers and timestamps:

{transcript_text}

Please generate a list of insightful and thoughtful questions the candidate could have asked the interviewer based on this discussion.
Make sure the questions reflect curiosity about the role, company, and expectations.
"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that suggests smart interview questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    questions = response.choices[0].message.content
    with open("suggested_questions.txt", "w", encoding="utf-8") as f:
        f.write(questions)
    return questions

#------------------------------- Gradio Interface


def process_media(file):
    transcript, json_path, plot_path, video_path = transcribe_video_or_audio(file)
    summary = generate_summary(transcript)
    questions = generate_questions(transcript)
    return transcript, json_path, plot_path, summary, questions, video_path

with gr.Blocks(css="""
    .video-preview-container {
        max-width: 400px !important;
        height: 250px !important;
        margin: 0 auto;
    }
    .video-preview-container video {
        width: 100% !important;
        height: 100% !important;
        border-radius: 8px;
        border: 2px solid #3498db;
        object-fit: contain;
    }
    .gr-button {
        font-weight: bold;
        border: 2px solid #2980b9 !important;
        background-color: #ecf0f1;
        color: #2c3e50;
    }
    .gr-button:hover {
        background-color: #d0e4f7 !important;
    }
    .gr-box, .gr-textbox, .gr-file, .gr-image {
        border: 2px solid #3498db !important;
        border-radius: 10px !important;
    }
    label {
        font-weight: bold !important;
        color: #2c3e50 !important;
    }
    #upload-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    #note {
        font-style: italic;
        color: #b71c1c;
        margin-top: 20px;
        font-size: 14px;
    }
    #developer {
        margin-top: 30px;
        font-weight: bold;
        color: #4e342e;
    }
    .section-divider {
        height: 2px;
        background-color: #cccccc;
        margin: 25px 0;
    }
    .upload-area {
        margin-bottom: 20px;
    }
""") as demo:
    gr.Markdown("# 🎧 Video/Audio Transcription and Speaker Analysis")

    with gr.Row():
        with gr.Column(scale=1, elem_id="upload-section"):
            gr.Markdown("### 📤 Upload Media")
            with gr.Column(elem_classes="upload-area"):
                media_input = gr.File(label="Video or Audio File", file_types=[".mp4", ".mov", ".avi", ".wav", ".mp3"])
                video_output = gr.Video(
                    label="Preview", 
                    interactive=False, 
                    visible=False, 
                    elem_classes="video-preview-container",
                    height=250  # ------------------------------- Explicit height setting
                )
            process_btn = gr.Button(" Process Media", variant="primary")
            gr.Markdown("**Note:** _Processing may take several minutes for longer files._", elem_id="note")

            gr.Markdown("---", elem_classes="section-divider")
            gr.Markdown("**👨‍💻 Developer:** Muhammad Shahjhan Gondal  \n📧 shahjhangondal99@gmail.com", elem_id="developer")

        with gr.Column(scale=3):
            gr.Markdown("------", elem_classes="section-divider")

            with gr.Tabs():
                with gr.Tab("📝 Transcript"):
                    transcript_output = gr.Textbox(label="Transcription", interactive=False, lines=20)

                with gr.Tab("🧠 Analysis"):
                    with gr.Row():
                        summary_output = gr.Textbox(label="Summary", interactive=False, lines=10)
                        questions_output = gr.Textbox(label="Suggested Questions", interactive=False, lines=10)

                    with gr.Row():
                        json_output = gr.File(label="Diarization JSON")
                        plot_output = gr.Image(label="Diarization Visualization", width=500)

    def update_video_display(file):
        if file and file.name.lower().endswith(('.mp4', '.mov', '.avi')):
            return gr.update(value=file.name, visible=True)
        return gr.update(visible=False)

    process_btn.click(
        fn=process_media,
        inputs=media_input,
        outputs=[transcript_output, json_output, plot_output, summary_output, questions_output, video_output]
    )

    media_input.change(fn=update_video_display, inputs=media_input, outputs=video_output)

    demo.launch()
