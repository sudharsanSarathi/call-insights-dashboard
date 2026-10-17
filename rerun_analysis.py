import os
import json
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

with open("output/Call Recording (1).mp3.json") as f:
    result = json.load(f)

entries = result.get("diarized_transcript", {}).get("entries", [])
transcript_lines = [
    f"[Speaker {e['speaker_id']}] ({e['start_time_seconds']:.1f}-{e['end_time_seconds']:.1f}s): {e['transcript']}"
    for e in entries
]
full_transcript_text = "\n".join(transcript_lines)

discovery_prompt = f"""You are analyzing a sales/tele-calling call transcript between an agent and a customer.
The transcript below has speakers labeled as Speaker 0 / Speaker 1 (not yet identified as agent/customer).
This is a Tamil-language real estate sales call.

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

Respond in clear structured JSON only, no extra commentary.
"""

response = client.chat.completions(
    model="sarvam-105b",
    messages=[{"role": "user", "content": discovery_prompt}],
    max_tokens=8000,
)
print(response.choices[0].message.content)
