from flask import Flask, request, jsonify, render_template

from dashboard_data import load_all_transcripts, build_dashboard, SYSTEM_PROMPT_TEMPLATE
from llm_client import complete, LLMError

app = Flask(__name__)


@app.route("/")
def index():
    try:
        dashboard = build_dashboard()
    except LLMError as e:
        dashboard = {
            "upcoming_visits": [], "common_demands": [], "top_properties": [],
            "agent_performance": {"best_agent": None, "worst_agent": None},
            "call_count": 0, "error": str(e),
        }
    return render_template("dashboard.html", dashboard=dashboard)


@app.route("/api/dashboard")
def dashboard():
    force_refresh = request.args.get("refresh") == "1"
    try:
        return jsonify(build_dashboard(force_refresh=force_refresh))
    except LLMError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json()
    history = body.get("history", [])
    user_message = body.get("message", "")

    transcripts_text = load_all_transcripts()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(transcripts=transcripts_text)

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        reply = complete(messages, max_tokens=8000, reasoning_effort="low") or "(empty response, please rephrase)"
    except LLMError as e:
        reply = f"⚠️ {e}"
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=False, port=5050)
