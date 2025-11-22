from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


def _playwright_launch_kwargs() -> Dict[str, Any]:
    """Build launch options for Playwright Chromium.

    Defaults are safe for headless Docker; dev scripts can override via
    PLAYWRIGHT_HEADLESS, PLAYWRIGHT_VIEWPORT_WIDTH/HEIGHT, PLAYWRIGHT_SLOW_MO_MS.
    """

    headless_env = os.environ.get("PLAYWRIGHT_HEADLESS", "true").lower()
    headless = headless_env not in {"0", "false", "no"}

    viewport_width = int(os.environ.get("PLAYWRIGHT_VIEWPORT_WIDTH", "1280"))
    viewport_height = int(os.environ.get("PLAYWRIGHT_VIEWPORT_HEIGHT", "720"))
    slow_mo = int(os.environ.get("PLAYWRIGHT_SLOW_MO_MS", "0"))

    return {
        "headless": headless,
        "viewport": {"width": viewport_width, "height": viewport_height},
        "slow_mo": slow_mo,
    }


def _normalize_site(site: Optional[str]) -> str:
    if not site:
        return "example.com"

    if "//" not in site:
        candidate = f"https://{site}"
    else:
        candidate = site

    parsed = urlparse(candidate)
    if parsed.netloc:
        return parsed.netloc
    if parsed.path:
        return parsed.path.lstrip("/")
    return site


def run_playwright_do_anything(
    *,
    site: Optional[str],
    steps: List[str],
    user_id: str,
    text: str,
    use_cached_path: bool,
    confirm: bool,
) -> Dict[str, Any]:
    """Generic Playwright runner for the /agents/do-anything agent.

    It opens the target site, takes sequential screenshots for each step, and
    applies very lightweight heuristics for search / checkout flows. For money
    steps it enforces an explicit confirm gate via `needs_approval`.
    """

    hostname = _normalize_site(site)
    base_url = f"https://{hostname}"

    user_data_dir = f"/tmp/playwright-do-anything/{user_id}/{hostname}"
    proofs_dir = str(Path(__file__).resolve().parent.parent.parent / "proofs")
    os.makedirs(proofs_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir,
            **_playwright_launch_kwargs(),
        )
        page = browser.new_page()

        ts = int(time.time())
        base = f"do_anything_{hostname}_{user_id}_{ts}"

        try:
            page.goto(base_url, wait_until="load")
            first_frame = os.path.join(proofs_dir, f"{base}_step0.png")
            page.screenshot(path=first_frame, full_page=True)

            # Very simple login detection: visible email field or generic sign-in
            # text. If detected, surface a structured login_required response.
            try:
                email_loc = page.locator("input[type='email'], input[name='email']").first
                signin_text = page.get_by_text("Sign in")
                needs_login = False
                try:
                    if email_loc.is_visible(timeout=2000):  # type: ignore[arg-type]
                        needs_login = True
                except PlaywrightTimeoutError:
                    pass
                except Exception:
                    pass

                if signin_text.count():
                    needs_login = True

                if needs_login:
                    browser.close()
                    return {
                        "status": "login_required",
                        "site": hostname,
                        "steps": steps,
                        "login_url": base_url,
                        "message": "User must log in to the target site once so cookies are saved.",
                        "proof_url": f"/proofs/{Path(first_frame).name}",
                    }
            except Exception:
                pass

            # Iterate through planned steps; for each step capture a screenshot
            # and apply a couple of simple heuristics.
            for idx, step in enumerate(steps or []):
                frame_path = os.path.join(proofs_dir, f"{base}_step{idx + 1}.png")

                lowered = step.lower()

                # Very naive search box interaction.
                if "search" in lowered:
                    query = step.split("search", 1)[1].strip(" :") or step
                    try:
                        search_box = page.get_by_role("searchbox").first
                    except Exception:
                        search_box = page.get_by_placeholder("Search").first
                    try:
                        if search_box:
                            search_box.click()
                            search_box.fill(query)
                            page.keyboard.press("Enter")
                            page.wait_for_timeout(3000)
                    except Exception:
                        pass

                # Money / checkout related steps → apply confirm gating.
                if any(k in lowered for k in ["checkout", "place order", "buy now", "pay"]):
                    if not confirm:
                        preview_filename = f"{base}_preview.png"
                        preview_path = os.path.join(proofs_dir, preview_filename)
                        page.screenshot(path=preview_path, full_page=True)
                        browser.close()
                        return {
                            "status": "needs_approval",
                            "site": hostname,
                            "steps": steps,
                            "proof_url": f"/proofs/{preview_filename}",
                        }

                    # Best-effort click of a primary buy/checkout button.
                    try:
                        for label in [
                            "Buy Now",
                            "Place your order",
                            "Proceed to checkout",
                            "Checkout",
                        ]:
                            btn = page.get_by_role("button", name=label)
                            if btn.count():
                                btn.first.click()
                                page.wait_for_timeout(3000)
                                break
                    except Exception:
                        pass

                page.screenshot(path=frame_path, full_page=True)

            final_filename = f"{base}_final.png"
            final_path = os.path.join(proofs_dir, final_filename)
            page.screenshot(path=final_path, full_page=True)

            browser.close()
            return {
                "status": "succeeded",
                "site": hostname,
                "steps": steps,
                "proof_url": f"/proofs/{final_filename}",
            }
        except Exception as exc:  # noqa: BLE001
            try:
                browser.close()
            except Exception:
                pass
            return {
                "status": "failed",
                "site": hostname,
                "steps": steps,
                "error": str(exc),
            }
