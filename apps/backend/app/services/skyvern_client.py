from typing import Any, Dict, Optional

import httpx
import time

from app.core.config import get_settings
from app.schemas import ElectricityProvider
from app.services.user_profiles import get_user_profile_for_provider


def _provider_portal_url(provider: ElectricityProvider) -> str:
    if provider == ElectricityProvider.bescom:
        return "https://www.bescom.co.in/"
    if provider == ElectricityProvider.mspdcl:
        return "https://www.mspdcl.co.in/"
    if provider == ElectricityProvider.tsspdcl:
        return "https://www.tssouthernpower.com/"
    return "https://www.google.com/"


def _build_instructions(
    provider: ElectricityProvider,
    phone_number: Optional[str],
    email: Optional[str],
    user_profile: Optional[Dict[str, str]],
    use_cached_path: bool,
) -> str:
    base = "You are Hemlo, an autonomous agent for paying Indian electricity bills. "
    portal = provider.value

    profile_bits: list[str] = []
    consumer_id = None
    profile_phone = None
    profile_email = None

    if user_profile:
        consumer_id = user_profile.get("consumer_id")
        profile_phone = user_profile.get("phone_number")
        profile_email = user_profile.get("email")

    if consumer_id:
        profile_bits.append(
            f"Use consumer number {consumer_id} from the user's saved profile."
        )
    if profile_phone:
        profile_bits.append(
            f"Prefer phone number {profile_phone} from the user's saved profile."
        )
    if profile_email:
        profile_bits.append(
            f"Prefer email {profile_email} from the user's saved profile."
        )

    if phone_number and phone_number != profile_phone:
        profile_bits.append(f"If needed, you may also use phone number {phone_number}.")
    if email and email != profile_email:
        profile_bits.append(f"If needed, you may also use email {email}.")

    profile_str = " ".join(profile_bits).strip()

    if use_cached_path:
        core = (
            "You have previously completed this exact electricity bill payment flow for this user on this portal. "
            "Reuse the same navigation path and button locations you learned before to reach the payment screen as fast as possible. "
            "Avoid exploring new pages unless something has changed. Once on the payment screen, pay the full outstanding electricity bill, "
            "wait for the final confirmation screen, and take a clear screenshot of the payment confirmation page. After payment is successful, stop."
        )
    else:
        core = (
            f"Open the official {portal} consumer portal, log in using the saved consumer details, "
            f"pay the full outstanding electricity bill using any available secure method, wait for the final confirmation screen, "
            f"and take a clear screenshot of the payment confirmation page. After payment is successful, stop."
        )

    if profile_str:
        return f"{base}{core} {profile_str}"
    return f"{base}{core}"


def run_electricity_payment_task(
    provider: ElectricityProvider,
    phone_number: Optional[str],
    email: Optional[str],
    user_id: str,
    use_cached_path: bool = False,
) -> Dict[str, Any]:
    settings = get_settings()
    base_url = settings.skyvern_base_url.rstrip("/")
    run_url = f"{base_url}/tasks/run"
    headers = {"x-api-key": "skyvern"}

    user_profile = get_user_profile_for_provider(user_id=user_id, provider=provider)
    instructions = _build_instructions(
        provider=provider,
        phone_number=phone_number,
        email=email,
        user_profile=user_profile,
        use_cached_path=use_cached_path,
    )
    url = _provider_portal_url(provider)

    payload: Dict[str, Any] = {
        "url": url,
        "instructions": instructions,
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(run_url, json=payload, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        task_id = data.get("task_id")
        if not isinstance(task_id, str):
            raise RuntimeError("Skyvern did not return a task_id")

        max_attempts = 60
        poll_interval = 5.0

        for _ in range(max_attempts):
            status_resp = client.get(f"{base_url}/tasks/{task_id}", headers=headers)
            status_resp.raise_for_status()
            task_data: Dict[str, Any] = status_resp.json()
            if task_data.get("status") == "completed":
                return task_data
            time.sleep(poll_interval)

    raise RuntimeError("Skyvern task did not complete in time")


def extract_screenshot_url(task_result: Dict[str, Any]) -> Optional[str]:
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
