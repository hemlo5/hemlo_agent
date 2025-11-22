from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from app.core.config import get_settings
from app.schemas import FoodPlatform


def _get_groq_client() -> Optional[OpenAI]:
    settings = get_settings()
    if not settings.groq_api_key:
        return None
    return OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")


def _get_sarvam_client() -> Optional[OpenAI]:
    settings = get_settings()
    if not settings.sarvam_api_key:
        return None
    # Assuming Sarvam exposes OpenAI-compatible endpoint; adjust base_url if needed.
    return OpenAI(api_key=settings.sarvam_api_key)


def _is_hindi_heavy(text: str) -> bool:
    hindi_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    return hindi_chars >= 0.7 * max(1, len(text))


def _choose_client_for_negotiation(text: str) -> tuple[OpenAI, str]:
    settings = get_settings()
    if _is_hindi_heavy(text) and settings.sarvam_api_key:
        client = _get_sarvam_client()
        if client is not None:
            return client, "sarvam"
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("No LLM client configured for refund negotiation")
    return client, "groq"


def generate_negotiation_message(text: str, platform: FoodPlatform, order_id: Optional[str]) -> str:
    """Use Llama 70B / Sarvam to generate a strong but polite refund request message."""

    client, provider = _choose_client_for_negotiation(text)

    system_prompt = (
        "You are Hemlo, an expert Indian customer support negotiator. "
        "Write a short, very convincing, but polite refund request message to Swiggy/Zomato support in Hinglish or Hindi, "
        "as appropriate from the user's text. Focus on issues like food being cold, late delivery, missing items, etc. "
        "Keep it under 4 sentences and avoid threats. Do not include any JSON, only the message body."
    )

    if platform == FoodPlatform.swiggy:
        platform_name = "Swiggy"
    else:
        platform_name = "Zomato"

    order_part = f"Order ID: {order_id}." if order_id else ""

    user_prompt = f"Platform: {platform_name}. {order_part} User request: {text!r}"

    response = client.chat.completions.create(
        model=get_settings().groq_llm_model_primary,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content or ""


def parse_refund_outcome(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort parse of refund outcome from Skyvern artifacts or metadata."""

    summary = raw_result.get("summary") or raw_result.get("metadata") or {}
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except json.JSONDecodeError:
            summary = {}

    refund_amount = None
    if isinstance(summary, dict):
        amt = summary.get("refund_amount")
        if isinstance(amt, (int, float)):
            refund_amount = float(amt)

    return {"refund_amount": refund_amount}
