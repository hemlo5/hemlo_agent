from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.core.config import get_settings


def parse_electricity_bill_request(text: str) -> Dict[str, Optional[str]]:
    """Use Groq Llama 3.1 70B to extract provider, phone, email, consumer_id.

    Returns a dict with keys: provider, phone, email, consumer_id.
    provider is the provider code (BESCOM/MSPDCL/TSSPDCL) or None.
    """

    settings = get_settings()
    if not settings.groq_api_key:
        return {"provider": None, "phone": None, "email": None, "consumer_id": None}

    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    system_prompt = (
        "You are a strict JSON API that extracts information from Indian electricity bill "
        "payment requests in Hindi or English. Return a JSON object with keys: provider, "
        "phone, email, consumer_id. provider must be one of: BESCOM, MSPDCL, TSSPDCL, or "
        "null if unknown. phone should be a mobile number string if present else null. "
        "email should be a single email address if present else null. consumer_id should "
        "be the electricity consumer/account number if present else null. Respond with "
        "JSON only."
    )

    user_prompt = f"Text: {text!r}"

    try:
        response = client.chat.completions.create(
            model=settings.groq_llm_model_primary,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        return {"provider": None, "phone": None, "email": None, "consumer_id": None}

    content = response.choices[0].message.content or "{}"

    try:
        data: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError:
        return {"provider": None, "phone": None, "email": None, "consumer_id": None}

    provider: Optional[str] = None
    raw_provider = data.get("provider")
    if isinstance(raw_provider, str):
        upper = raw_provider.strip().upper()
        if upper in {"BESCOM", "MSPDCL", "TSSPDCL"}:
            provider = upper

    phone: Optional[str] = data.get("phone") if isinstance(data.get("phone"), str) else None
    email: Optional[str] = data.get("email") if isinstance(data.get("email"), str) else None
    consumer_id: Optional[str] = (
        data.get("consumer_id") if isinstance(data.get("consumer_id"), str) else None
    )

    return {
        "provider": provider,
        "phone": phone,
        "email": email,
        "consumer_id": consumer_id,
    }


def parse_stripe_refund_request(text: str) -> Dict[str, Optional[Any]]:
    settings = get_settings()
    if not settings.groq_api_key:
        return {
            "charge_id": None,
            "amount_cents": None,
            "currency": None,
            "reason": None,
        }

    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    system_prompt = (
        "You are a strict JSON API that extracts information from Stripe refund "
        "requests. Return a JSON object with keys: charge_id, amount_cents, "
        "currency, reason. charge_id is the Stripe charge id if present (for "
        "example 'ch_...'), else null. amount_cents is the amount to refund in "
        "the smallest currency unit as an integer (for USD/EUR, cents), or null "
        "if the full amount should be refunded. currency is the 3-letter ISO "
        "code in lowercase (for example 'usd', 'eur') or null. reason is a "
        "short human-readable reason for the refund or null. Respond with JSON "
        "only."
    )

    user_prompt = f"Text: {text!r}"

    try:
        response = client.chat.completions.create(
            model=settings.groq_llm_model_primary,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        return {
            "charge_id": None,
            "amount_cents": None,
            "currency": None,
            "reason": None,
        }

    content = response.choices[0].message.content or "{}"

    try:
        data: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError:
        return {
            "charge_id": None,
            "amount_cents": None,
            "currency": None,
            "reason": None,
        }

    charge_id: Optional[str] = None
    raw_charge_id = data.get("charge_id")
    if isinstance(raw_charge_id, str):
        charge_id = raw_charge_id.strip() or None

    amount_cents: Optional[int] = None
    raw_amount = data.get("amount_cents")
    if isinstance(raw_amount, (int, float)):
        amount_cents = int(raw_amount)

    currency: Optional[str] = None
    raw_currency = data.get("currency")
    if isinstance(raw_currency, str):
        cur = raw_currency.strip().lower()
        if len(cur) == 3:
            currency = cur

    reason: Optional[str] = None
    raw_reason = data.get("reason")
    if isinstance(raw_reason, str):
        cleaned = raw_reason.strip()
        if cleaned:
            reason = cleaned

    return {
        "charge_id": charge_id,
        "amount_cents": amount_cents,
        "currency": currency,
        "reason": reason,
    }


def parse_do_anything_request(text: str) -> Dict[str, Any]:
    """Parse a free-form natural language request into a generic site + steps plan.

    The model is expected to return JSON of the form:

    {"site": "amazon.in", "steps": ["search iPhone 15", "sort by price", ...]}
    """

    settings = get_settings()
    if not settings.groq_api_key:
        return {"site": None, "steps": []}

    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    system_prompt = (
        "You are a strict JSON API that plans automation flows for any website. "
        "Given a natural language request, return a JSON object with keys: "
        "site, steps. 'site' should be a hostname like 'amazon.in', 'stripe.com', "
        "'twitter.com', 'linkedin.com', or null if unknown. 'steps' must be an "
        "ordered array of short strings, each describing one high-level step the "
        "agent should perform in the browser or via APIs. Respond with JSON only."
    )

    user_prompt = f"Text: {text!r}"

    try:
        response = client.chat.completions.create(
            model=settings.groq_llm_model_primary,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        return {"site": None, "steps": []}

    content = response.choices[0].message.content or "{}"
    try:
        data: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError:
        return {"site": None, "steps": []}

    site: Optional[str] = None
    raw_site = data.get("site")
    if isinstance(raw_site, str):
        cleaned = raw_site.strip()
        site = cleaned or None

    steps: List[str] = []
    raw_steps = data.get("steps")
    if isinstance(raw_steps, list):
        for item in raw_steps:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    steps.append(s)

    return {"site": site, "steps": steps}
