import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import demucs.pretrained
from demucs.apply import apply_model
import whisper
from moviepy import VideoFileClip

# path setup and output dir
VIDEO_PATH = f"data/input_video.mp4"
MODEL_NAME = "htdemucs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "vocal_segments"
AUDIO_EXTRACTED = "temp_audio.wav"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Device: {DEVICE}")

def extract_audio(video_path, output_audio):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio, codec='pcm_s16le')
    return output_audio

audio_path = extract_audio(VIDEO_PATH, AUDIO_EXTRACTED)

# load demucs for separation and silero for vad
print("Loading Demucs separation model")
demucs_model = demucs.pretrained.get_model(MODEL_NAME)
demucs_model.to(DEVICE)
demucs_model.eval()

print("Loading Silero VAD model")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
(get_speech_timestamps, _, read_audio, *_) = utils

# load wav file and convert to tensor
print(f"Loading extracted audio from: {audio_path}")
audio_data, sample_rate = sf.read(audio_path)
if audio_data.ndim == 1:
    audio_data = np.expand_dims(audio_data, axis=0)
else:
    audio_data = audio_data.T

waveform = torch.from_numpy(audio_data).float().to(DEVICE)

# resample if input rate doesn't match demucs
if sample_rate != demucs_model.samplerate:
    print(f"Resampling for Demucs: {sample_rate} -> {demucs_model.samplerate}")
    resampler = torchaudio.transforms.Resample(sample_rate, demucs_model.samplerate).to(DEVICE)
    waveform = resampler(waveform)
    sample_rate = demucs_model.samplerate

# run inference to extract vocal stem
print("Running Demucs separation")
with torch.no_grad():
    sources = apply_model(demucs_model, waveform.unsqueeze(0))[0]

# index 3 is the vocal track
vocals = sources[3] 

# silero needs 16khz mono for detection
VAD_SR = 16000
print(f"Preparing vocals for VAD (resampling to {VAD_SR}Hz)")
vocal_mono = torch.mean(vocals, dim=0, keepdim=True) 
resampler_vad = torchaudio.transforms.Resample(sample_rate, VAD_SR).to(DEVICE)
vocal_vad_input = resampler_vad(vocal_mono)

# find speech timestamps with 0.5 sec silence buffer
print("Detecting speech segments")
speech_timestamps = get_speech_timestamps(vocal_vad_input.cpu(), vad_model, 
                                          sampling_rate=VAD_SR,
                                          threshold=0.6, 
                                          min_silence_duration_ms=500)

# slice audio based on timestamps and export files
print(f"Found {len(speech_timestamps)} segments. Saving and transcribing")
print("Loading Whisper model")
whisper_model = whisper.load_model("base").to(DEVICE)

transcripts = []

for i, ts in enumerate(speech_timestamps):
    # rescale timestamps back to original sample rate
    start_sample = int(ts['start'] * (sample_rate / VAD_SR))
    end_sample = int(ts['end'] * (sample_rate / VAD_SR))
    
    segment = vocals[:, start_sample:end_sample].cpu()
    
    segment_path = os.path.join(OUTPUT_DIR, f"vocal_instance_{i+1:03d}.wav")
    torchaudio.save(segment_path, segment, sample_rate)

    ## whisper transcription
    result = whisper_model.transcribe(segment_path, language="en") 
    text = result['text'].strip()
    
    transcripts.append({
        "id": i+1,
        "start": ts['start'] / VAD_SR,
        "end": ts['end'] / VAD_SR,
        "text": text
    })
    print(f"[{i+1}] {text}")

print(f"Successfully saved {len(speech_timestamps)} files to '{OUTPUT_DIR}'")

with open("transcriere.txt", "w", encoding="utf-8") as f:
    for entry in transcripts:
        f.write(f"[{entry['start']:.2f}s - {entry['end']:.2f}s]: {entry['id']:.2f} : {entry['text']}\n")

# helper function to visualize frequency data
def plot_spectrogram(wave, title):
    specgram = torchaudio.transforms.Spectrogram()(wave.cpu())
    log_spec = 10 * torch.log10(specgram[0] + 1e-9)
    plt.imshow(log_spec.numpy(), aspect="auto", origin="lower")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")

# compare original audio vs extracted vocals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_spectrogram(waveform, "Original Mixture")
plt.subplot(2, 1, 2)
plot_spectrogram(vocals, "Full Isolated Vocals")
plt.tight_layout()
plt.show()