from app.celery_app import celery_app
from app.services.path_cache import is_path_cached, mark_path_cached
from app.services.playwright_amazon_order import run_playwright_amazon_order_cart
from app.services.playwright_do_anything import run_playwright_do_anything
from app.services.playwright_stripe_refunds import run_playwright_stripe_refund
from app.services.stripe_refunds import create_stripe_refund


@celery_app.task(name="agents.stripe_refund")
def stripe_refund_task(payload: dict) -> dict:
    charge_id = payload.get("charge_id")
    if not charge_id:
        return {
            "status": "failed",
            "error": "missing_charge_id",
        }

    amount_cents = payload.get("amount_cents")
    currency = payload.get("currency")
    reason = payload.get("reason")
    use_api = bool(payload.get("use_api"))
    confirm = bool(payload.get("confirm"))

    # For now we use a fixed demo user id; later this can come from auth/session.
    user_id = payload.get("user_id") or "demo-user"

    # Simple Redis-backed path cache keyed by (user, flow="stripe_refund").
    flow_key = "stripe_refund"
    use_cached_path = is_path_cached(user_id=user_id, provider=flow_key)

    if not use_api:
        # Default: Playwright + Chromium against stripe.com dashboard.
        result = run_playwright_stripe_refund(
            charge_id=charge_id,
            amount_cents=amount_cents,
            currency=currency,
            reason=reason,
            user_id=user_id,
            use_cached_path=use_cached_path,
            confirm=confirm,
        )

        status = result.get("status")
        if status == "succeeded":
            # Mark path cached after the first successful interactive run.
            mark_path_cached(user_id=user_id, provider=flow_key)
            return {
                "status": "succeeded",
                "charge_id": charge_id,
                "refund_id": result.get("refund_id"),
                "amount_cents": result.get("amount_cents"),
                "currency": result.get("currency"),
                "reason": result.get("reason"),
                "proof_url": result.get("proof_url"),
                "raw_result": result,
            }

        if status == "login_required":
            # Let the client handle login UX explicitly instead of silently
            # falling back to API.
            return result

        if status == "needs_approval":
            # Surface preview-only state; do not fall back to API.
            return result

        # If Playwright path fails for other reasons, we fall back to Stripe API
        # if available (power users / servers with STRIPE_SECRET_KEY).

    # API fallback path (or forced when use_api=True). Respect confirm flag so
    # the client can implement an Approve/Deny step even for pure API refunds.
    if not confirm:
        return {
            "status": "needs_approval",
            "charge_id": charge_id,
            "amount_cents": amount_cents,
            "currency": currency,
            "reason": reason,
            "message": "Set confirm=true to actually create the refund via Stripe API.",
        }

    refund = create_stripe_refund(
        charge_id=charge_id,
        amount_cents=amount_cents,
        reason=reason,
    )

    refund_id = refund.get("id")
    amount = refund.get("amount")
    final_currency = currency or refund.get("currency")
    final_reason = reason or refund.get("reason")

    proof_url = None
    if isinstance(refund_id, str):
        proof_url = f"https://dashboard.stripe.com/refunds/{refund_id}"

    return {
        "status": "succeeded",
        "charge_id": charge_id,
        "refund_id": refund_id,
        "amount_cents": amount,
        "currency": final_currency,
        "reason": final_reason,
        "proof_url": proof_url,
        "raw_result": refund,
    }


@celery_app.task(name="agents.amazon_order_cart")
def amazon_order_cart_task(payload: dict) -> dict:
    # For now we use a fixed demo user id; later this can come from auth/session.
    user_id = payload.get("user_id") or "demo-user"
    confirm = bool(payload.get("confirm"))

    flow_key = "amazon_cart"
    use_cached_path = is_path_cached(user_id=user_id, provider=flow_key)

    result = run_playwright_amazon_order_cart(
        user_id=user_id,
        use_cached_path=use_cached_path,
        confirm=confirm,
    )

    status = result.get("status")
    if status == "succeeded":
        mark_path_cached(user_id=user_id, provider=flow_key)
        return {
            "status": "succeeded",
            "order_id": result.get("order_id"),
            "proof_url": result.get("proof_url"),
            "raw_result": result,
        }

    if status == "login_required":
        return result

    if status == "needs_approval":
        return result

    return {
        "status": status or "failed",
        "error": result.get("error"),
        "raw_result": result,
    }


@celery_app.task(name="agents.do_anything")
def do_anything_task(payload: dict) -> dict:
    text = payload.get("text") or ""
    confirm = bool(payload.get("confirm"))

    user_id = payload.get("user_id") or "demo-user"

    # Very simple routing: for now we only support the generic Playwright path.
    site = payload.get("site")
    steps = payload.get("steps") or []

    flow_key = f"do_anything:{site or 'unknown'}"
    use_cached_path = is_path_cached(user_id=user_id, provider=flow_key)

    result = run_playwright_do_anything(
        site=site,
        steps=steps,
        user_id=user_id,
        text=text,
        use_cached_path=use_cached_path,
        confirm=confirm,
    )

    status = result.get("status")
    if status == "succeeded":
        mark_path_cached(user_id=user_id, provider=flow_key)
        return {
            "status": "succeeded",
            "site": result.get("site"),
            "steps": result.get("steps"),
            "proof_url": result.get("proof_url"),
            "raw_result": result,
        }

    if status in {"login_required", "needs_approval"}:
        return result

    return {
        "status": status or "failed",
        "error": result.get("error"),
        "site": result.get("site"),
        "steps": result.get("steps"),
        "raw_result": result,
    }
