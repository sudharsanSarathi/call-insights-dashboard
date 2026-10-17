from dashboard_data import load_all_transcripts, SYSTEM_PROMPT_TEMPLATE
from llm_client import complete, PROVIDER


def main():
    transcripts_text = load_all_transcripts()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(transcripts=transcripts_text)
    history = [{"role": "system", "content": system_prompt}]

    print(f"Chat with your call transcripts (provider: {PROVIDER}). Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message.lower() in ("exit", "quit"):
            break

        history.append({"role": "user", "content": user_message})

        reply = complete(history, max_tokens=8000, reasoning_effort="low") or "(empty response, please rephrase)"
        history.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
