import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("transformers")
install("torch")
install("torchaudio")
install("librosa")
install("TTS")
install("gradio")

import torch
import librosa
import tempfile
import gradio as gr
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

# Load Dysarthria Whisper ASR Model
asr_model = WhisperForConditionalGeneration.from_pretrained("wh1tewhale/dysarthria-automatic-speech-recognition").to("cuda" if torch.cuda.is_available() else "cpu")
asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")  # Using base processor

# Load Grammar Correction (optional)
gc_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
gc_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Load Coqui TTS
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

def transcribe_and_synthesize(audio_path):
    if audio_path is None:
        return None, "No input received."

    # 1. Load audio
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

    # 2. Transcribe
    inputs = asr_processor(audio=speech_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(asr_model.device)

    with torch.no_grad():
        generated_ids = asr_model.generate(input_features)

    transcription = asr_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 3. Synthesize clean audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tts.tts_to_file(text=transcription, file_path=tmp_wav.name)
        tts_audio_path = tmp_wav.name

    return tts_audio_path, transcription
def transcribe_and_synthesize(audio_path):
    if audio_path is None:
        return None, "No input received."

    # 1. Load audio
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

    # 2. Transcribe
    inputs = asr_processor(audio=speech_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(asr_model.device)

    with torch.no_grad():
        generated_ids = asr_model.generate(input_features)

    transcription = asr_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 3. Grammar correction
    input_text = f"Correct this sentence: {transcription}"
    inputs = gc_tokenizer(input_text, return_tensors="pt").to(gc_model.device)
    outputs = gc_model.generate(**inputs)
    corrected_text = gc_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 4. TTS synthesis on corrected text
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tts.tts_to_file(text=corrected_text, file_path=tmp_wav.name)
        tts_audio_path = tmp_wav.name

    return tts_audio_path, corrected_text


gr.Interface(
    fn=transcribe_and_synthesize,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),
    outputs=[
        gr.Audio(label="Synthesized Audio", autoplay=False),
        gr.Textbox(label="Grammar Corrected Text", lines=3)
    ],
    title="Dysarthric Speech Transcriber + Respeaker",
    description="🎤 Upload or record speech. Returns grammar-corrected and synthesized speech."
).launch()
