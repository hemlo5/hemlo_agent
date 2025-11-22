import os
import sys
import json
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright
from openai import OpenAI


PROOFS_DIR = Path("proofs")
PROOFS_DIR.mkdir(parents=True, exist_ok=True)


def _get_groq_client() -> OpenAI | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[WARN] GROQ_API_KEY not set; skipping LLM parsing.")
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def parse_stripe_refund_text(text: str) -> dict:
    client = _get_groq_client()
    if client is None:
        return {"charge_id": None, "amount_cents": None, "currency": None, "reason": None}

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

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text: {text!r}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Groq call failed: {exc}")
        return {"charge_id": None, "amount_cents": None, "currency": None, "reason": None}

    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {"charge_id": None, "amount_cents": None, "currency": None, "reason": None}

    result: dict = {
        "charge_id": None,
        "amount_cents": None,
        "currency": None,
        "reason": None,
    }
    if isinstance(data.get("charge_id"), str):
        result["charge_id"] = data["charge_id"].strip() or None
    if isinstance(data.get("amount_cents"), (int, float)):
        result["amount_cents"] = int(data["amount_cents"])
    if isinstance(data.get("currency"), str):
        cur = data["currency"].strip().lower()
        if len(cur) == 3:
            result["currency"] = cur
    if isinstance(data.get("reason"), str):
        cleaned = data["reason"].strip()
        if cleaned:
            result["reason"] = cleaned
    return result


def run_amazon_demo() -> None:
    user_data_dir = str(Path(".playwright-amazon-demo").resolve())
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            slow_mo=800,
        )
        page = browser.new_page()
        try:
            # "load" is more reliable than strict networkidle here.
            page.goto("https://www.amazon.in/gp/cart/view.html", wait_until="load")
            ts = "amazon_final"
            proof_path = PROOFS_DIR / f"{ts}.png"

            # If not logged in, pause and let the user complete login once.
            try:
                email_loc = page.locator("input[name='email']").first
                signin_btn = page.get_by_role("button", name="Sign in to your account")
                signin_text = page.get_by_text("Sign in to your account")

                needs_login = False
                try:
                    if email_loc.is_visible(timeout=2000):  # type: ignore[arg-type]
                        needs_login = True
                except Exception:
                    pass

                if signin_btn.count() or signin_text.count():
                    needs_login = True

                if needs_login:
                    # Best-effort: click the CTA for the user if present.
                    try:
                        if signin_btn.count():
                            signin_btn.first.click()
                        elif signin_text.count():
                            signin_text.first.click()
                    except Exception:
                        pass

                    print("[INFO] You are not logged in to Amazon yet.")
                    print("      In the visible browser window, complete the Amazon login flow.")
                    print("      Once you see your cart with items, press Enter here to continue...")
                    input()
                    page.goto("https://www.amazon.in/gp/cart/view.html", wait_until="load")
            except Exception:
                pass

            # Try to click a generic checkout / proceed button a few times.
            for _ in range(5):
                # Prefer Proceed to checkout / Checkout / Place your order.
                for label in [
                    "Proceed to checkout",
                    "Proceed to Buy",
                    "Checkout",
                    "Place your order",
                ]:
                    btn = page.get_by_role("button", name=label)
                    if btn.count():
                        btn.first.click()
                        page.wait_for_timeout(3000)
                        break
                else:
                    break

            page.wait_for_timeout(5000)
            page.screenshot(path=str(proof_path), full_page=True)
            print(f"Done! Amazon demo proof: {proof_path}")
            print("[INFO] Browser will stay open for your recording.")
            input("      When you're done, press Enter here to close it...")
        finally:
            browser.close()


def run_agent_demo() -> None:
    text = input("Describe what Hemlo should do (any site, any task): ")
    confirm = input("Auto-approve money actions? (y/N): ").strip().lower().startswith("y")

    client = httpx.Client(base_url="http://localhost:8000", timeout=10.0)
    try:
        resp = client.post("/agents/do-anything", json={"text": text, "confirm": confirm})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to create do-anything task: {exc}")
        return

    data = resp.json()
    task_id = data.get("task_id")
    if not task_id:
        print(f"[ERROR] Unexpected response: {data}")
        return

    print(f"[INFO] Created do-anything task: {task_id}")

    while True:
        try:
            time.sleep(2)
            status_resp = client.get(f"/agents/do-anything/{task_id}")
            status_resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Poll failed: {exc}")
            return

        payload = status_resp.json()
        status_value = payload.get("status")
        print(f"[INFO] Task status: {status_value}")

        if status_value == "running":
            continue

        result = payload.get("result") or {}
        proof_url = result.get("proof_url")
        site = result.get("site")
        steps = result.get("steps") or []

        print(f"Final status: {status_value}")
        if site:
            print(f"Site: {site}")
        if steps:
            print("Steps:")
            for s in steps:
                print(f"  - {s}")
        if proof_url:
            print(f"Proof URL: {proof_url}")
        break


def run_stripe_demo() -> None:
    text = input("Describe the refund (e.g. 'refund $20 for the last charge'): ")
    parsed = parse_stripe_refund_text(text)
    print(f"[INFO] LLM parsed: {parsed}")

    user_data_dir = str(Path(".playwright-stripe-demo").resolve())
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            slow_mo=800,
        )
        page = browser.new_page()
        try:
            page.goto("https://dashboard.stripe.com/payments", wait_until="networkidle")

            # If not logged in, ask the user to log in once.
            try:
                email_loc = page.locator("input[type='email']").first
                if email_loc.is_visible(timeout=2000):  # type: ignore[arg-type]
                    print("[INFO] Please log in to Stripe Dashboard in the visible browser window.")
                    print("      Once you see the Payments list, press Enter here to continue...")
                    input()
            except Exception:
                pass

            # Very simple: if we have a charge_id, paste it into the search box.
            charge_id = parsed.get("charge_id")
            if charge_id:
                try:
                    search_box = page.get_by_placeholder("Search").first
                    search_box.click()
                    search_box.fill(charge_id)
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(4000)
                except Exception:
                    pass

            # This is intentionally loose: in the recording you can manually take over
            # with the visible browser if the locator fails.
            try:
                refund_button = page.get_by_role("button", name="Refund").first
                if refund_button:
                    refund_button.click()
                    page.wait_for_timeout(2000)
            except Exception:
                print("[WARN] Could not auto-click Refund button; you can click it manually in the browser.")
                input("Press Enter here after you complete the refund manually...")

            proof_path = PROOFS_DIR / "stripe_refund_final.png"
            page.wait_for_timeout(3000)
            page.screenshot(path=str(proof_path), full_page=True)
            print(f"Done! Stripe demo proof: {proof_path}")
            print("[INFO] Browser will stay open for your recording.")
            input("      When you're done, press Enter here to close it...")
        finally:
            browser.close()


def main() -> None:
    mode = input("Type 'amazon', 'stripe', or 'agent': ").strip().lower()
    if mode == "amazon":
        run_amazon_demo()
    elif mode == "stripe":
        run_stripe_demo()
    elif mode in {"agent", "anything"}:
        run_agent_demo()
    else:
        print("Unknown mode. Please run again and type 'amazon' or 'stripe'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
