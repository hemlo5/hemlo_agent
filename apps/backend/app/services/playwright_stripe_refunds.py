from __future__ import annotations

from typing import Any, Dict, Optional

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
from pathlib import Path
import os
import time


def run_playwright_stripe_refund(
    *,
    charge_id: str,
    amount_cents: Optional[int],
    currency: Optional[str],
    reason: Optional[str],
    user_id: str,
    use_cached_path: bool,
    confirm: bool,
) -> Dict[str, Any]:
    """Best-effort Stripe refund via Chromium + Playwright.

    Uses a per-user persistent context directory so login cookies are reused
    across runs. If the user is not logged in, this will return
    {"status": "login_required", ...} so the app can instruct the user to
    log in once in a real browser / guided flow.
    """

    user_data_dir = f"/tmp/playwright-stripe/{user_id}"
    proofs_dir = str(Path(__file__).resolve().parent.parent.parent / "proofs")
    os.makedirs(proofs_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            slow_mo=800,
        )
        page = browser.new_page()

        try:
            # Go directly to Payments. If not logged in, Stripe will redirect
            # to login.
            page.goto("https://dashboard.stripe.com/payments", wait_until="networkidle")

            # First frame for live streaming.
            ts = int(time.time())
            first_frame = os.path.join(proofs_dir, f"stripe_refund_{charge_id}_{ts}_step1.png")
            page.screenshot(path=first_frame, full_page=True)

            # Detect login requirement by checking for an email field.
            if page.locator("input[type='email']").first.is_visible(timeout=2000):  # type: ignore[arg-type]
                return {
                    "status": "login_required",
                    "login_url": "https://dashboard.stripe.com/login",
                    "message": "User must log in to Stripe dashboard once so cookies are saved.",
                }

            # Very rough heuristic: use search to locate the charge, then open it.
            # Selectors are text/placeholder-based to be more robust, but may
            # still need tuning against the live Stripe dashboard.
            try:
                search_box = page.get_by_placeholder("Search payments")
            except PlaywrightTimeoutError:
                search_box = None

            if search_box:
                search_box.fill(charge_id)
                search_box.press("Enter")
                page.wait_for_timeout(3000)

                second_frame = os.path.join(proofs_dir, f"stripe_refund_{charge_id}_{ts}_step2.png")
                page.screenshot(path=second_frame, full_page=True)

            # Click the first row that mentions the charge id.
            page.get_by_text(charge_id, exact=False).first.click()
            page.wait_for_timeout(2000)

            third_frame = os.path.join(proofs_dir, f"stripe_refund_{charge_id}_{ts}_step3.png")
            page.screenshot(path=third_frame, full_page=True)

            # Look for a Refund button and click it.
            page.get_by_text("Refund").first.click()
            page.wait_for_timeout(1000)

            # If a specific amount is requested, try to fill it.
            if amount_cents is not None:
                amount_str = str(amount_cents / 100.0)
                amount_input = page.get_by_role("textbox", name="Amount", exact=False)
                if amount_input:
                    amount_input.fill(amount_str)

            # At this point we are on the refund confirmation UI. If confirm=False,
            # capture a preview and stop so the client can show an Approve/Deny UX.
            preview_filename = f"stripe_refund_{charge_id}_{ts}_preview.png"
            preview_path = os.path.join(proofs_dir, preview_filename)
            page.screenshot(path=preview_path, full_page=True)

            if not confirm:
                browser.close()
                return {
                    "status": "needs_approval",
                    "charge_id": charge_id,
                    "amount_cents": amount_cents,
                    "currency": currency,
                    "reason": reason,
                    "proof_url": f"/proofs/{preview_filename}",
                }

            # Confirm the refund.
            page.get_by_role("button", name="Refund").first.click()

            # Wait a bit for confirmation UI.
            page.wait_for_timeout(4000)

            # Take final screenshot as proof into the shared proofs volume.
            final_filename = f"stripe_refund_{charge_id}_{int(time.time())}_final.png"
            screenshot_path = os.path.join(proofs_dir, final_filename)
            page.screenshot(path=screenshot_path, full_page=True)

            browser.close()

            return {
                "status": "succeeded",
                "charge_id": charge_id,
                "amount_cents": amount_cents,
                "currency": currency,
                "reason": reason,
                "proof_url": f"/proofs/{final_filename}",
            }
        except Exception as exc:  # noqa: BLE001
            browser.close()
            return {
                "status": "failed",
                "error": str(exc),
            }
