import os
import json
import hashlib
import difflib
import re

from llm_client import complete

PROCESSED_DIR = "processed"
INSIGHTS_PATH = os.path.join(PROCESSED_DIR, "_insights.json")

SYSTEM_PROMPT_TEMPLATE = """You are an assistant with access to a set of tele-calling sales call summaries.
Each call has already been analyzed and structured into fields like agent name, customer name, property,
budget, scheduled visit, customer requirements, issues, and outcome.

Answer the user's questions using ONLY the information in these call summaries. Be specific, cite the call
name (e.g. "Call Recording (3).mp3") when relevant, and say clearly if something is not present in the
summaries rather than guessing.

When referring to a speaker, never say "Speaker 0" or "Speaker 1". Use the agent_name field for the agent
(or "Agent" if it is null), and use the customer_name field for the customer written as "Customer (Name)"
(or just "Customer" if it is null).

Be precise and direct. If the question asks for a specific target (e.g. "who performed well", "which
property is most popular", "when is the next visit"), give ONLY that answer — one name/value, or at most
two if there is a genuine tie — with a single short reason or supporting fact. Do not list or rank every
agent, call, or property unless the user explicitly asks for a full list or comparison. If the user asks
a broad/open question, a short summary is fine, but still prefer the fewest words that fully answer it.

Do not use markdown formatting — no "**bold**", no "#" headings, no markdown bullet characters. Write in
plain sentences. If a list is genuinely needed, use plain numbered lines like "1. ..." on separate lines,
nothing fancier.

CALL SUMMARIES:
{transcripts}
"""


def load_processed_calls():
    calls = []
    if not os.path.isdir(PROCESSED_DIR):
        return calls
    for filename in sorted(os.listdir(PROCESSED_DIR)):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        with open(os.path.join(PROCESSED_DIR, filename)) as f:
            calls.append(json.load(f))
    return calls


def load_all_transcripts():
    calls = load_processed_calls()
    blocks = []
    for call in calls:
        blocks.append(f"=== CALL: {call.get('call_name')} ===\n" + json.dumps(call, ensure_ascii=False, indent=2))
    return "\n\n".join(blocks)


def _calls_signature(calls):
    names = sorted(c.get("call_name", "") for c in calls)
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()


def _load_cached_insights(calls):
    if not os.path.exists(INSIGHTS_PATH):
        return None
    with open(INSIGHTS_PATH) as f:
        cached = json.load(f)
    if cached.get("signature") != _calls_signature(calls):
        return None
    return cached


def _save_insights(calls, common_demands, agent_performance):
    payload = {
        "signature": _calls_signature(calls),
        "common_demands": common_demands,
        "agent_performance": agent_performance,
    }
    with open(INSIGHTS_PATH, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _day_sort_key(day):
    lower = (day or "").lower()
    if "today" in lower:
        return (0, lower)
    if "tomorrow" in lower:
        return (1, lower)
    for i, wd in enumerate(_WEEKDAYS):
        if wd in lower:
            return (2 + i, lower)
    return (100, lower)


def _time_sort_key(time_str):
    if not time_str:
        return (9999, time_str or "")
    match = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", time_str, re.IGNORECASE)
    if not match:
        return (9999, time_str)
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return (hour * 60 + minute, time_str)


def _compute_upcoming_visits(calls):
    by_day = {}
    for call in calls:
        visit = call.get("scheduled_visit") or {}
        day = visit.get("day")
        time = visit.get("time")
        if not day and not time:
            continue
        day_label = day or "Unspecified day"
        buyer = call.get("customer_name") or call.get("buyer_summary") or "Customer"
        site = call.get("primary_site_name") or call.get("location") or "Unknown site"
        agent = call.get("agent_name") or "Agent"

        entry = by_day.setdefault(day_label, [])
        entry.append({
            "call_name": call.get("call_name"),
            "buyer": buyer,
            "site": site,
            "agent": agent,
            "time": time,
            "confirmed": bool(visit.get("confirmed")),
        })

    grouped = []
    for day_label, visits in by_day.items():
        visits.sort(key=lambda v: _time_sort_key(v["time"]))
        grouped.append({"day": day_label, "visits": visits})

    grouped.sort(key=lambda g: _day_sort_key(g["day"]))
    return grouped


def _compute_top_properties(calls):
    ranked = []
    for group in _group_properties(calls).values():
        scores = [c.get("customer_interest_score") or 0 for c in group["calls"]]
        mentions = len(group["calls"])
        avg_score = round(sum(scores) / mentions, 2) if mentions else 0
        ranked.append({"property": group["display_name"], "avg_score": avg_score, "mentions": mentions})
    ranked.sort(key=lambda e: (e["mentions"], e["avg_score"]), reverse=True)
    return ranked[:5]


def _normalize_name_key(name):
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


# ASR/LLM extraction phrases the same real agent or site differently across calls
# (e.g. "Banu Priya" / "Bhanupriya" / "Panupriya"; "Aniyapuram Parali Division Site" /
# "Aniyapuram (Parali Division)"), so exact-match grouping undercounts. Cluster by
# fuzzy similarity instead. Person names tolerate more spelling drift than place
# names (which often share generic words like "Road"/"Project" across unrelated
# sites), so properties use a stricter threshold to avoid false merges.
AGENT_SIMILARITY_THRESHOLD = 0.6
PROPERTY_SIMILARITY_THRESHOLD = 0.7


def _fuzzy_cluster(calls, name_fn, threshold):
    clusters = []  # list of {"key": normalized_rep, "names": [...], "calls": [...]}
    for call in calls:
        name = name_fn(call)
        if not name:
            continue
        key = _normalize_name_key(name)

        best_cluster = None
        best_ratio = 0.0
        for cluster in clusters:
            ratio = difflib.SequenceMatcher(None, key, cluster["key"]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_cluster = cluster

        if best_cluster is not None and best_ratio >= threshold:
            best_cluster["names"].append(name)
            best_cluster["calls"].append(call)
        else:
            clusters.append({"key": key, "names": [name], "calls": [call]})

    groups = {}
    for cluster in clusters:
        display_name = max(set(cluster["names"]), key=cluster["names"].count)
        groups[cluster["key"]] = {"display_name": display_name, "calls": cluster["calls"]}
    return groups


def _group_agents(calls):
    return _fuzzy_cluster(calls, lambda c: c.get("agent_name") or "Unknown", AGENT_SIMILARITY_THRESHOLD)


def _group_properties(calls):
    return _fuzzy_cluster(calls, lambda c: c.get("primary_site_name"), PROPERTY_SIMILARITY_THRESHOLD)


def _call_llm_for_insights(calls):
    demands_input = [c.get("customer_requirements") or [] for c in calls]
    agents_input = {}
    for group in _group_agents(calls).values():
        display_name = group["display_name"]
        agents_input.setdefault(display_name, [])
        for call in group["calls"]:
            agents_input[display_name].append({
                "call_name": call.get("call_name"),
                "outcome": call.get("outcome"),
                "customer_interest_score": call.get("customer_interest_score"),
                "issues": call.get("issues") or [],
            })

    prompt = f"""You are analyzing aggregated data from several real-estate tele-sales calls.

CUSTOMER REQUIREMENTS PER CALL:
{json.dumps(demands_input, ensure_ascii=False, indent=2)}

PER-AGENT CALL DATA:
{json.dumps(agents_input, ensure_ascii=False, indent=2)}

Respond with STRICT JSON only, no markdown fences, no commentary, matching this shape exactly:
{{
  "common_demands": ["bullet point 1", "bullet point 2", ...],
  "best_agent": {{"name": "...", "reason": "..."}},
  "worst_agent": {{"name": "...", "reason": "..."}}
}}

"common_demands" should be a short bulleted synthesis (4-6 bullets) of what customers most commonly ask
about or care about across all calls. "best_agent" must name exactly ONE agent (from the per-agent data)
who performed best, with a concrete reason. "worst_agent" must name exactly one agent who had the most
issues/hiccups or worst outcomes, with a concrete reason citing what went wrong. All text values (bullets
and reasons) must be plain sentences with no markdown — no "**bold**", no "#" headings, no markdown bullet
characters."""

    text = complete([{"role": "user", "content": prompt}], max_tokens=16000, reasoning_effort="low").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def build_dashboard(force_refresh=False):
    calls = load_processed_calls()

    cached = None if force_refresh else _load_cached_insights(calls)
    if cached is not None:
        common_demands = cached["common_demands"]
        agent_performance = cached["agent_performance"]
    else:
        insights = _call_llm_for_insights(calls)
        common_demands = insights.get("common_demands", [])
        agent_performance = {
            "best_agent": insights.get("best_agent"),
            "worst_agent": insights.get("worst_agent"),
        }
        _save_insights(calls, common_demands, agent_performance)

    return {
        "upcoming_visits": _compute_upcoming_visits(calls),
        "common_demands": common_demands,
        "top_properties": _compute_top_properties(calls),
        "agent_performance": agent_performance,
        "call_count": len(calls),
    }
