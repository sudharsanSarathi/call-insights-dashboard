import os
import json

from llm_client import complete

# ============================================================================
# TODO — BEFORE THE NEXT REPROCESSING RUN, add these to the extraction schema:
#
# 1. Timestamped issues/highlights: change "issues" from a list of plain
#    strings to a list of {"time": "...", "quote": "...", "description": "..."}
#    objects — "quote" being a short verbatim excerpt from the transcript so
#    the chat/dashboard can cite real evidence instead of abstract reasoning.
#    Add a parallel "highlights" field (same shape) for positive moments
#    (what the agent did well), not just problems.
#
# 2. Audio overlap detection: NOT an extraction/LLM change — compute directly
#    from transcripts/*.json by finding diarized entries from different
#    speakers whose [start_time_seconds, end_time_seconds] ranges overlap.
#    Belongs in dashboard_data.py (or a new helper), not this file, but note
#    it here since it was requested alongside the above.
#
# Once added: update the chat system prompt and the insights-generation
# prompt in dashboard_data.py to actually cite these quotes/timestamps when
# answering "why"/reasoning questions (e.g. "who performed well").
#
# Requested 2026-10-15; deferred to a future reprocessing pass.
# ============================================================================

TRANSCRIPTS_DIR = "transcripts"
PROCESSED_DIR = "processed"

EXTRACTION_PROMPT = """You are analyzing a real-estate tele-sales call transcript. Speakers are labeled
Speaker 0 / Speaker 1 (not yet identified as agent/customer). The call may be in Tamil, English, or mixed.

Transcript:
{transcript}

Figure out which speaker is the sales agent and which is the customer. Then respond with STRICT JSON only,
no markdown fences, no commentary, matching this shape exactly:
{{
  "agent_name": "string or null, the agent's name if ever stated in the call",
  "customer_name": "string or null, the customer's name if ever stated in the call",
  "buyer_summary": "max 6 words describing the buyer when their name is unknown, e.g. 'Namakkal local, own-use buyer'. Use 'Customer' if nothing distinctive is known.",
  "primary_site_name": "max 6 words, the ONE main site/project the customer is actually planning to visit or is most interested in (not every location mentioned in passing). Romanize if the call is in Tamil, e.g. 'Aniyapuram Site (Parali Division)'.",
  "property_names": ["list of distinct site/property names or identifiers mentioned, for reference only"],
  "location": "short description of the area/road/landmarks discussed",
  "budget_range": "string describing price/budget discussed, or null",
  "customer_requirements": ["short phrases describing what the customer asked about or cares about"],
  "scheduled_visit": {{"day": "string or null, e.g. Monday", "time": "string or null, e.g. 10:00 AM", "confirmed": true/false}},
  "customer_interest_score": "integer 1-5, how interested/positive the customer seemed",
  "issues": ["short phrases describing anything the agent could not answer, confusion, or friction points"],
  "outcome": "one of: visit_scheduled, callback_requested, no_interest, undetermined"
}}
"""


def build_transcript_text(data):
    entries = data.get("diarized_transcript", {}).get("entries", [])
    lines = [
        f"[Speaker {e['speaker_id']}] ({e['start_time_seconds']:.1f}-{e['end_time_seconds']:.1f}s): {e['transcript']}"
        for e in entries
    ]
    return "\n".join(lines)


def extract_structured(transcript_text):
    prompt = EXTRACTION_PROMPT.format(transcript=transcript_text)
    text = complete([{"role": "user", "content": prompt}], max_tokens=16000, reasoning_effort="low").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filenames = sorted(f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".json"))

    for filename in filenames:
        call_name = filename.replace(".json", "")
        out_path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(out_path):
            print(f"[skip] {call_name} already processed")
            continue

        print(f"Processing {call_name}...")
        with open(os.path.join(TRANSCRIPTS_DIR, filename)) as f:
            data = json.load(f)
        transcript_text = build_transcript_text(data)

        try:
            structured = extract_structured(transcript_text)
        except Exception as e:
            print(f"  FAILED to process {call_name}: {e!r}")
            continue

        structured["call_name"] = call_name
        out_path = os.path.join(PROCESSED_DIR, filename)
        with open(out_path, "w") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
