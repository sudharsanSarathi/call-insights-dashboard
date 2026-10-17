import os
import json
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
API_KEY = os.environ["SARVAM_API_KEY"]

AUDIO_DIR = "audios"
OUTPUT_DIR = "transcripts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = SarvamAI(api_subscription_key=API_KEY)

audio_files = sorted(
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"))
)
print(f"Found {len(audio_files)} audio files to transcribe.\n")

for filename in audio_files:
    out_path = os.path.join(OUTPUT_DIR, filename + ".json")
    if os.path.exists(out_path):
        print(f"[skip] {filename} already transcribed")
        continue

    src_path = os.path.join(AUDIO_DIR, filename)
    print(f"[start] {filename}")
    try:
        job = client.speech_to_text_job.create_job(
            model="saaras:v4",
            mode="transcribe",
            language_code="unknown",
            with_diarization=True,
            num_speakers=2,
        )
        job.upload_files(file_paths=[src_path])
        job.start()
        job.wait_until_complete(poll_interval=5, timeout=900)
        job.download_outputs(output_dir=OUTPUT_DIR)
        print(f"[done]  {filename}")
    except Exception as e:
        print(f"[FAILED] {filename}: {e!r}")

print("\nAll files processed. Transcripts saved in ./transcripts/")
