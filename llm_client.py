import os
import httpx
from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError as SarvamApiError

load_dotenv()


class LLMError(Exception):
    """Raised with a short, user-facing message when the active LLM provider can't serve a request."""


def _friendly_error(provider, status_code):
    if status_code == 402:
        return f"{provider} has run out of credits. Top up the account or switch LLM_PROVIDER in .env."
    if status_code == 429:
        return f"{provider} is rate-limiting requests right now. Wait a bit and try again, or switch LLM_PROVIDER in .env."
    return f"{provider} request failed (HTTP {status_code})."

PROVIDER = os.environ.get("LLM_PROVIDER", "sarvam").lower()

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
SARVAM_MODEL = os.environ.get("SARVAM_MODEL", "sarvam-105b")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_sarvam_client = None


def _get_sarvam_client():
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    return _sarvam_client


def _complete_sarvam(messages, max_tokens, reasoning_effort):
    try:
        response = _get_sarvam_client().chat.completions(
            model=SARVAM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except SarvamApiError as e:
        raise LLMError(_friendly_error("Sarvam", e.status_code)) from e
    message = response.choices[0].message
    return message.content or message.reasoning_content or ""


def _complete_groq(messages, max_tokens, reasoning_effort):
    response = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        },
        timeout=120,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise LLMError(_friendly_error("Groq", response.status_code)) from e
    message = response.json()["choices"][0]["message"]
    return message.get("content") or message.get("reasoning") or ""


def complete(messages, max_tokens=4000, reasoning_effort="low"):
    """Returns the assistant's reply text, regardless of which provider (LLM_PROVIDER env var) is active."""
    if PROVIDER == "groq":
        return _complete_groq(messages, max_tokens, reasoning_effort)
    return _complete_sarvam(messages, max_tokens, reasoning_effort)
