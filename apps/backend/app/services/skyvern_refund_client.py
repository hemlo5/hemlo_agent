from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings
from app.schemas import FoodPlatform


def _platform_refund_url(platform: FoodPlatform) -> str:
    if platform == FoodPlatform.swiggy:
        return "https://www.swiggy.com/my-account/orders"
    return "https://www.zomato.com/orders"


def run_refund_negotiation_task(
    platform: FoodPlatform,
    order_id: Optional[str],
    negotiation_message: str,
) -> Dict[str, Any]:
    settings = get_settings()
    base_url = settings.skyvern_base_url.rstrip("/")
    run_url = f"{base_url}/tasks/run"
    headers = {"x-api-key": "skyvern"}

    url = _platform_refund_url(platform)

    instructions = (
        "You are Hemlo, negotiating a refund with Swiggy/Zomato for the user. "
        "Open the orders page, locate the relevant order (using the order ID if provided), open the help/support/chat section, "
        "and use the following message (paraphrased if needed) to request a refund: "
        f"'{negotiation_message}'. "
        "Try to secure the maximum fair refund or credits available. Once the platform confirms a refund or compensation, "
        "take a clear screenshot of the confirmation page or chat and then stop."
    )

    payload: Dict[str, Any] = {
        "url": url,
        "instructions": instructions,
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(run_url, json=payload, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        task_id = data.get("task_id")
        if not isinstance(task_id, str):
            raise RuntimeError("Skyvern did not return a task_id for refund negotiation")

        # For MVP we assume Skyvern returns final task in the same response or shortly via polling
        status_resp = client.get(f"{base_url}/tasks/{task_id}", headers=headers)
        status_resp.raise_for_status()
        task_data: Dict[str, Any] = status_resp.json()
        return task_data


def extract_refund_screenshot_url(task_result: Dict[str, Any]) -> Optional[str]:
    artifacts = task_result.get("artifacts") or []
    if not isinstance(artifacts, list) or not artifacts:
        return None
    first = artifacts[0]
    if not isinstance(first, dict):
        return None
    url = first.get("file_url")
    if isinstance(url, str):
        return url
    return None
