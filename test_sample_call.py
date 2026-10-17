import os
import sys
import json
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

API_KEY = os.environ.get("SARVAM_API_KEY")
if not API_KEY:
    sys.exit("SARVAM_API_KEY not found in .env")

AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else "audios/Call Recording (1).mp3"

client = SarvamAI(api_subscription_key=API_KEY)

print(f"Submitting batch STT job for: {AUDIO_PATH}")
job = client.speech_to_text_job.create_job(
    model="saaras:v4",
    mode="transcribe",
    language_code="unknown",
    with_diarization=True,
    num_speakers=2,
)
job.upload_files(file_paths=[AUDIO_PATH])
job.start()
print("Job started, polling for completion...")
job.wait_until_complete(poll_interval=5, timeout=600)

out_dir = "output"
os.makedirs(out_dir, exist_ok=True)
job.download_outputs(output_dir=out_dir)

result_files = [f for f in os.listdir(out_dir) if f.endswith(".json")]
if not result_files:
    sys.exit("No output file found after job completion.")

result_path = os.path.join(out_dir, sorted(result_files)[-1])
with open(result_path) as f:
    result = json.load(f)

print("\n=== RAW TRANSCRIPT ===")
print(result.get("transcript", "(no transcript field)"))

entries = result.get("diarized_transcript", {}).get("entries", [])
print("\n=== DIARIZED TRANSCRIPT ===")
transcript_lines = []
for e in entries:
    line = f"[Speaker {e['speaker_id']}] ({e['start_time_seconds']:.1f}-{e['end_time_seconds']:.1f}s): {e['transcript']}"
    print(line)
    transcript_lines.append(line)

full_transcript_text = "\n".join(transcript_lines) or result.get("transcript", "")

discovery_prompt = f"""You are analyzing a sales/tele-calling call transcript between an agent and a customer.
The transcript below has speakers labeled as Speaker 0 / Speaker 1 (not yet identified as agent/customer).

Your task: analyze this call and list ALL the categories, parameters, or tags you can extract from it that would be
useful to track consistently across many similar sales calls. Do not limit yourself to a predefined list - think
freely about what is extractable and meaningful for sales call quality analysis, customer intent, and outcomes.

For each category/tag you propose, give:
- tag_name (short, reusable label)
- description (what it captures)
- value_for_this_call (what you observed in this specific transcript)

Also identify:
- which speaker (0 or 1) is the sales agent vs the customer, and how you inferred it
- the agent's name if mentioned in the transcript
- the conclusion/outcome of this call (e.g. did the customer agree to visit, ask for a callback, say no, etc.)

Transcript:
{full_transcript_text}

Respond in clear structured JSON.
"""

print("\n=== CALLING SARVAM LLM FOR TAG DISCOVERY ===")
response = client.chat.completions(
    model="sarvam-105b",
    messages=[{"role": "user", "content": discovery_prompt}],
    max_tokens=8000,
)
print(response.choices[0].message.content)
