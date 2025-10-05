import os
import subprocess
import shutil
import uvicorn
import traceback
import time
import asyncio
import sys
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import moviepy as mp
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import edge_tts
import ffmpeg
import torch

# **FIX:** Add the OpenVoice directory to the Python path to make its modules importable
sys.path.append('OpenVoice')

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# --- FastAPI Setup ---
app = FastAPI()
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Model Loading (Global for efficiency) ---
print("Loading OpenVoice models...")
base_speaker_tts = None
tone_color_converter = None
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the base model for synthesizing speech in different languages
    base_speaker_tts = BaseSpeakerTTS(f'checkpoints/base_speakers/EN/config.json', device=device)
    base_speaker_tts.load_ckpt(f'checkpoints/base_speakers/EN/en_base_speaker_tts.pth')

    # Load the tone color converter for voice cloning
    tone_color_converter = ToneColorConverter('checkpoints/converter/config.json', device=device)
    tone_color_converter.load_ckpt('checkpoints/converter/converter.pth')
    
    print("✅ OpenVoice models loaded successfully and are ready.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load the OpenVoice models on startup. Voice cloning will be unavailable.")
    print(f"Error details: {e}")

# --- API Endpoints ---
@app.post("/process-video/")
async def create_upload_file(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    language: str = Form(...),
    subtitles: bool = Form(...),
    tts_model: str = Form("gtts")
):
    temp_dir = "temp_uploads"; os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, video.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(video.file, buffer)
    background_tasks.add_task(process_video_task, file_path, language, subtitles, tts_model)
    return {"message": f"Processing started for {video.filename}."}

# --- Core Processing Functions ---

def run_openvoice_tts(text, speaker_wav_path, language, output_path):
    """Uses the globally loaded OpenVoice model for voice cloning."""
    global device # Access the global device variable
    if not base_speaker_tts or not tone_color_converter:
        raise RuntimeError("OpenVoice models are not available. Check server startup logs.")

    print(f"Running OpenVoice cloning using sample: {speaker_wav_path}")
    
    # Extract the tone color embedding from the reference audio
    target_se, audio_name = se_extractor.get_se(speaker_wav_path, tone_color_converter, target_dir='temp/processed_se', vad=True)

    # Use a generic speaker based on the language
    # OpenVoice uses a standard set of speakers. 'EN-Default' is a good generic one.
    source_se = torch.load(f'checkpoints/base_speakers/EN/en_default_se.pth', map_location=device)

    # Synthesize the base audio in the target language
    save_path = f'{output_path}.tmp.wav'
    base_speaker_tts.tts(text, save_path, speaker='EN-Default', language=language.upper(), speed=1.0)
    
    # Convert the tone color of the synthesized audio to match the reference
    tone_color_converter.convert(
        audio_src_path=save_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=output_path,
        message="Encoding source audio...")
    
    if os.path.exists(save_path):
        os.remove(save_path) # Clean up temporary file
    print("-> OpenVoice cloning complete.")

# ... (Wav2Lip and other functions remain the same) ...
def run_wav2lip(video_path, audio_path, output_path):
    print("Wav2Lip is no longer used. Skipping this step.")
async def text_to_speech_edge(text, output_audio, voice):
    communicate = edge_tts.Communicate(text, voice); await communicate.save(output_audio)
def text_to_speech_gtts(text, output_audio, lang):
    tts = gTTS(text=text, lang=lang); tts.save(output_audio)
def extract_audio(video_path, audio_output):
    print("Step 1/8: Extracting audio...");
    with mp.VideoFileClip(video_path) as video: video.audio.write_audiofile(audio_output, logger=None);
    print("-> Audio extracted.")
def transcribe_audio_segments(audio_path):
    print("Step 2/8: Transcribing audio to segments...");
    model = whisper.load_model("base"); result = model.transcribe(audio_path);
    print("-> Transcription complete."); return result["segments"]
def translate_text(text, tgt_lang):
    print(f"Translating to language code: {tgt_lang}")
    model_name = f'Helsinki-NLP/opus-mt-en-{tgt_lang}';
    tokenizer = MarianTokenizer.from_pretrained(model_name); model = MarianMTModel.from_pretrained(model_name);
    tokens = tokenizer.encode(text, return_tensors="pt"); translated_tokens = model.generate(tokens);
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
def create_subtitle_file_from_segments(segments, output_path):
    print("Step 7/8: Creating subtitles...");
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            start, end, text = segment['start'], segment['end'], segment['translated_text']
            f.write(f"{i+1}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n");
    print(f"-> Subtitle file created at: {output_path}")
def format_time(seconds):
    hours, rem = divmod(seconds, 3600); minutes, seconds = divmod(rem, 60);
    millis = int((seconds - int(seconds)) * 1000);
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{millis:03d}"
def add_subtitles_to_video(video_path, output_path, subtitle_path):
    print("Step 8/8: Adding subtitles to video...");
    input_video = ffmpeg.input(video_path)
    escaped_path = subtitle_path.replace('\\', '/').replace(':', '\\:')
    video_stream = ffmpeg.filter(input_video.video, 'subtitles', filename=escaped_path)
    output_stream = ffmpeg.output(video_stream, input_video.audio, output_path, vcodec='libx264', acodec='copy')
    ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
    print("-> Subtitles added.")

# --- Main Task Function ---
def process_video_task(video_path: str, lang_code: str, include_subtitles: bool, tts_model: str):
    asyncio.run(async_process_video_task(video_path, lang_code, include_subtitles, tts_model))

async def async_process_video_task(video_path, lang_code, include_subtitles, tts_model):
    temp_dir = "temp"
    try:
        original_base, _ = os.path.splitext(os.path.basename(video_path));
        safe_base = "".join(c for c in original_base if c.isalnum() or c in ('-', '_')).rstrip();
        output_dir = "outputs"; os.makedirs(output_dir, exist_ok=True);
        final_video_path = os.path.join(output_dir, f"{original_base}_{lang_code}_dubbed.mp4");
        os.makedirs(temp_dir, exist_ok=True);
        extracted_audio_path = os.path.join(temp_dir, f"{safe_base}_audio.wav")
        final_dubbed_audio_path = os.path.join(temp_dir, f"{safe_base}_dubbed_audio.mp3");
        lipsync_output_path = os.path.join(temp_dir, f"{safe_base}_lipsync.mp4")
        
        extract_audio(video_path, extracted_audio_path)
        print("Step 3&4/8: Translating text for each segment...");
        segments = transcribe_audio_segments(extracted_audio_path);
        for seg in segments: seg['translated_text'] = translate_text(seg['text'], lang_code);
        
        print(f"Step 5/8: Generating speech using {tts_model.upper()}...");
        translated_audio_clips = []
        for i, seg in enumerate(segments):
            tts_path = os.path.join(temp_dir, f"seg_{i}.wav"); 
            
            if tts_model == 'openvoice_clone':
                run_openvoice_tts(seg['translated_text'], extracted_audio_path, lang_code, tts_path)
            elif tts_model == 'edge_tts':
                voice_map = {
                    'hi': 'hi-IN-MadhurNeural', 'ta': 'ta-IN-PallaviNeural', 'te': 'te-IN-MohanNeural',
                    'ml': 'ml-IN-MidhunNeural', 'kn': 'kn-IN-GaganNeural', 'bn': 'bn-IN-BashkarNeural',
                    'mr': 'mr-IN-ManoharNeural',
                }
                voice = voice_map.get(lang_code, 'en-US-AriaNeural')
                await text_to_speech_edge(seg['translated_text'], tts_path, voice=voice);
            else:
                tts_path_mp3 = os.path.join(temp_dir, f"seg_{i}.mp3")
                text_to_speech_gtts(seg['translated_text'], tts_path_mp3, lang=lang_code);
                tts_path = tts_path_mp3

            translated_audio_clips.append(mp.AudioFileClip(tts_path).with_start(seg['start']));
        
        final_audio = None
        with mp.VideoFileClip(video_path) as video:
            final_audio = mp.CompositeAudioClip(translated_audio_clips);
            final_audio.duration = video.duration;
            final_audio.write_audiofile(final_dubbed_audio_path, logger=None);
        
        if final_audio: final_audio.close()
        for clip in translated_audio_clips: clip.close()

        # Replace the audio in the video with the dubbed audio
        print("Step 6/8: Replacing video audio with dubbed audio...");
        with mp.VideoFileClip(video_path) as video_clip:
            dubbed_video_clip = video_clip.with_audio(mp.AudioFileClip(final_dubbed_audio_path))
            dubbed_video_path = os.path.join(temp_dir, f"{safe_base}_dubbed_video.mp4")
            dubbed_video_clip.write_videofile(dubbed_video_path, codec='libx264', audio_codec='aac', logger=None)
        print("-> Audio replaced in video.")

        # Wav2Lip is no longer used, so use the dubbed video for output
        lipsync_output_path = dubbed_video_path

        if include_subtitles:
            subtitle_file = os.path.join(temp_dir, f"{safe_base}_subtitles.srt");
            create_subtitle_file_from_segments(segments, subtitle_file);
            add_subtitles_to_video(lipsync_output_path, final_video_path, subtitle_file);
        else: shutil.copy(lipsync_output_path, final_video_path)

        print("✅ Processing complete!")
    except Exception as e:
        print(f"❌ An error occurred: {e}"); traceback.print_exc()
    finally:
        time.sleep(2);
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir);
        upload_dir = os.path.dirname(video_path)
        if os.path.exists(upload_dir): shutil.rmtree(upload_dir);

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000)

