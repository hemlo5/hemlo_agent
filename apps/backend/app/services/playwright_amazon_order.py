from __future__ import annotations

from typing import Any, Dict

from playwright.sync_api import sync_playwright
from pathlib import Path
import os
import time


def run_playwright_amazon_order_cart(*, user_id: str, use_cached_path: bool, confirm: bool) -> Dict[str, Any]:
    """Order the most recent thing in the user's Amazon cart via Chromium.

    This uses a persistent user_data_dir per user so that Amazon login cookies
    are reused across runs. If the user is not logged in, it returns a
    login_required status so the client can guide the user through first-time
    login.
    """

    user_data_dir = f"/tmp/playwright-amazon/{user_id}"
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
            page.goto("https://www.amazon.in/gp/cart/view.html", wait_until="networkidle")

            ts = int(time.time())
            base = f"amazon_order_{user_id}_{ts}"
            # First live frame from the cart.
            frame1 = os.path.join(proofs_dir, f"{base}_step1.png")
            page.screenshot(path=frame1, full_page=True)

            # Very simple login detection: look for an email field.
            email_loc = page.locator("input[name='email']").first
            if email_loc.is_visible(timeout=2000):  # type: ignore[arg-type]
                return {
                    "status": "login_required",
                    "login_url": "https://www.amazon.com/ap/signin",
                    "message": "User must log in to Amazon once so cookies are saved.",
                }

            # Click "Proceed to checkout" from the cart.
            page.get_by_role("button", name="Proceed to checkout").first.click()
            page.wait_for_timeout(3000)

            frame2 = os.path.join(proofs_dir, f"{base}_step2.png")
            page.screenshot(path=frame2, full_page=True)

            # Keep clicking the main yellow button (e.g. Continue / Place your order)
            # a few times until we reach the confirmation page.
            for step_idx in range(5):
                frame_path = os.path.join(proofs_dir, f"{base}_step{step_idx + 3}.png")
                page.screenshot(path=frame_path, full_page=True)

                # Try a "Place your order" button first.
                place_btn = page.get_by_role("button", name="Place your order")
                if place_btn.count():
                    # If confirm=False, stop here and surface a preview instead of
                    # actually placing the order.
                    preview_filename = f"{base}_preview.png"
                    preview_path = os.path.join(proofs_dir, preview_filename)
                    page.screenshot(path=preview_path, full_page=True)

                    if not confirm:
                        browser.close()
                        return {
                            "status": "needs_approval",
                            "order_id": None,
                            "proof_url": f"/proofs/{preview_filename}",
                        }

                    place_btn.first.click()
                    page.wait_for_timeout(4000)
                    break

                # Otherwise click a generic "Continue" button if present.
                cont_btn = page.get_by_role("button", name="Continue")
                if cont_btn.count():
                    cont_btn.first.click()
                    page.wait_for_timeout(3000)
                else:
                    break

            # Wait for confirmation UI.
            page.wait_for_timeout(5000)

            # Best-effort extraction of order number.
            order_number = None
            try:
                el = page.get_by_text("Order #").first
                text = el.inner_text()
                if "#" in text:
                    order_number = text.split("#")[-1].strip()
            except Exception:
                try:
                    el = page.get_by_text("Order number").first
                    text = el.inner_text()
                    parts = text.split()
                    if parts:
                        order_number = parts[-1].strip()
                except Exception:
                    order_number = None

            final_filename = f"{base}_final.png"
            screenshot_path = os.path.join(proofs_dir, final_filename)
            page.screenshot(path=screenshot_path, full_page=True)

            browser.close()

            return {
                "status": "succeeded",
                "order_id": order_number,
                "proof_url": f"/proofs/{final_filename}",
            }
        except Exception as exc:  # noqa: BLE001
            browser.close()
            return {
                "status": "failed",
                "error": str(exc),
            }
