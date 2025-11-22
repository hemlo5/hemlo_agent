from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings


def create_stripe_refund(
    charge_id: str,
    amount_cents: Optional[int],
    reason: Optional[str],
) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.stripe_secret_key:
        raise RuntimeError("STRIPE_SECRET_KEY is not configured")

    headers = {
        "Authorization": f"Bearer {settings.stripe_secret_key}",
    }

    data: Dict[str, Any] = {"charge": charge_id}
    if amount_cents is not None:
        data["amount"] = int(amount_cents)
    if reason:
        data["reason"] = reason

    with httpx.Client(timeout=30) as client:
        response = client.post("https://api.stripe.com/v1/refunds", data=data, headers=headers)
        response.raise_for_status()
        refund_data: Dict[str, Any] = response.json()

    return refund_data
