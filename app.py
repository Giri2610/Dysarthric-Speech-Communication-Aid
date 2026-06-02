🚀 STEP 1 — INSTALL LIBRARIES (RUN FIRST)
!pip install -q transformers librosa gradio soundfile sentencepiece accelerate gtts
!pip install -q torch torchaudio

🧠 STEP 2 — IMPORT LIBRARIES
import torch
import librosa
import gradio as gr
import tempfile

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from gtts import gTTS

⚙️ STEP 3 — LOAD MODELS
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Whisper ASR model (dysarthric speech)
asr_model = WhisperForConditionalGeneration.from_pretrained(
    "wh1tewhale/dysarthria-automatic-speech-recognition"
).to(device)

asr_processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small"
)

# FLAN-T5 LLM for correction
gc_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
).to(device)

gc_tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

🔥 STEP 4 — MAIN PIPELINE FUNCTION
def speech_pipeline(audio_path):

    if audio_path is None:
        return None, "No audio uploaded"

    # Load audio (16kHz required for Whisper)
    speech, sr = librosa.load(audio_path, sr=16000)

    # --------------------------
    # 1. Whisper ASR
    # --------------------------
    inputs = asr_processor(
        audio=speech,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = asr_model.generate(inputs["input_features"])

    transcription = asr_processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    # --------------------------
    # 2. FLAN-T5 Correction
    # --------------------------
    prompt = f"Correct this sentence: {transcription}"

    gc_inputs = gc_tokenizer(prompt, return_tensors="pt", truncation=True)
    gc_inputs = {k: v.to(device) for k, v in gc_inputs.items()}

    with torch.no_grad():
        outputs = gc_model.generate(**gc_inputs, max_new_tokens=64)

    corrected_text = gc_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # --------------------------
    # 3. Text-to-Speech (gTTS)
    # --------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts = gTTS(text=corrected_text, lang="en")
        tts.save(f.name)
        audio_output = f.name

    return audio_output, corrected_text

🎤 STEP 5 — GRADIO UI (RUN THIS LAST)
demo = gr.Interface(
    fn=speech_pipeline,
    inputs=gr.Audio(type="filepath", label="Upload / Record Speech"),
    outputs=[
        gr.Audio(label="Generated Speech Output"),
        gr.Textbox(label="Corrected Text")
    ],
    title="LLM-Based Dysarthric Speech Communication Aid",
    description="Whisper ASR + FLAN-T5 Correction + TTS (gTTS)"
)

demo.launch()
