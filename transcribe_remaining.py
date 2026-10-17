import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
API_KEY = os.environ["SARVAM_API_KEY"]

AUDIO_DIR = "audios"
OUTPUT_DIR = "transcripts"
MAX_WORKERS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
client = SarvamAI(api_subscription_key=API_KEY)


def transcribe_one(filename):
    src_path = os.path.join(AUDIO_DIR, filename)
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
        return filename, True, None
    except Exception as e:
        return filename, False, repr(e)


def main():
    audio_files = sorted(
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"))
    )
    targets = [
        f for f in audio_files
        if not os.path.exists(os.path.join(OUTPUT_DIR, f + ".json"))
    ]

    print(f"{len(targets)} file(s) left to transcribe (of {len(audio_files)} total). Running {MAX_WORKERS} in parallel.\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(transcribe_one, f): f for f in targets}
        for future in as_completed(futures):
            filename, ok, error = future.result()
            if ok:
                print(f"[done]   {filename}")
            else:
                print(f"[FAILED] {filename}: {error}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
