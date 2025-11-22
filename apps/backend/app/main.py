from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.celery_app import celery_app
from app.schemas import (
    AmazonOrderCartRequest,
    AmazonOrderCartResult,
    AmazonOrderCartTaskCreateResponse,
    AmazonOrderCartTaskStatusResponse,
    DoAnythingRequest,
    DoAnythingResult,
    DoAnythingTaskCreateResponse,
    DoAnythingTaskStatusResponse,
    OtpSubmitRequest,
    OtpSubmitResponse,
    StripeRefundRequest,
    StripeRefundResult,
    StripeRefundTaskCreateResponse,
    StripeRefundTaskStatusResponse,
)
from app.services.llm_parsing import parse_do_anything_request, parse_stripe_refund_request
from app.services.otp_store import save_otp


app = FastAPI(title="Hemlo AI Backend")


PUBLIC_BASE_URL = "http://localhost:8000"


def _build_public_proof_url(proof_path: str | None) -> str | None:
    if not proof_path:
        return None
    if proof_path.startswith("http://") or proof_path.startswith("https://"):
        return proof_path
    if not proof_path.startswith("/"):
        proof_path = "/" + proof_path
    return f"{PUBLIC_BASE_URL}{proof_path}"


proofs_dir = Path(__file__).resolve().parent.parent / "proofs"
proofs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/proofs", StaticFiles(directory=str(proofs_dir)), name="proofs")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/device/otp", response_model=OtpSubmitResponse)
def submit_otp(body: OtpSubmitRequest) -> OtpSubmitResponse:
    channel = body.channel or "sms"
    save_otp(user_id=body.user_id, otp=body.otp, channel=channel)
    return OtpSubmitResponse(status="stored")


@app.post("/agents/stripe-refund", response_model=StripeRefundTaskCreateResponse)
def create_stripe_refund_task(body: StripeRefundRequest) -> StripeRefundTaskCreateResponse:
    parsed = parse_stripe_refund_request(body.text)

    charge_id_from_llm = parsed.get("charge_id")
    amount_cents_from_llm = parsed.get("amount_cents")
    currency_from_llm = parsed.get("currency")
    reason_from_llm = parsed.get("reason")

    charge_id = body.charge_id or charge_id_from_llm
    amount_cents = body.amount_cents if body.amount_cents is not None else amount_cents_from_llm
    currency = body.currency or currency_from_llm
    reason = body.reason or reason_from_llm

    if not charge_id:
        raise HTTPException(status_code=400, detail="Unable to determine Stripe charge_id from request")

    payload = {
        "charge_id": charge_id,
        "amount_cents": amount_cents,
        "currency": currency,
        "reason": reason,
        "use_api": body.use_api,
        "confirm": body.confirm,
        "user_id": "demo-user",
    }

    task = celery_app.send_task("agents.stripe_refund", args=[payload])
    return StripeRefundTaskCreateResponse(task_id=task.id, status="queued")


@app.get("/agents/stripe-refund/{task_id}", response_model=StripeRefundTaskStatusResponse)
def get_stripe_refund_status(task_id: str) -> StripeRefundTaskStatusResponse:
    async_result = AsyncResult(task_id, app=celery_app)

    if async_result.state in {"PENDING", "RECEIVED", "STARTED"}:
        return StripeRefundTaskStatusResponse(
            status="running",
            message="Processing Stripe refund",
            result=None,
        )

    if async_result.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(async_result.result))

    if async_result.state != "SUCCESS":
        raise HTTPException(status_code=500, detail=f"Unexpected task state: {async_result.state}")

    data = async_result.result or {}

    proof_url = _build_public_proof_url(data.get("proof_url"))

    result = StripeRefundResult(
        status=data.get("status", "unknown"),
        charge_id=data.get("charge_id"),
        refund_id=data.get("refund_id"),
        amount_cents=data.get("amount_cents"),
        currency=data.get("currency"),
        reason=data.get("reason"),
        proof_url=proof_url,
        raw_result=data.get("raw_result"),
    )

    message = "Done!" if result.status == "succeeded" else "Something went wrong"

    return StripeRefundTaskStatusResponse(status=result.status, message=message, result=result)


@app.post("/agents/amazon-order-cart", response_model=AmazonOrderCartTaskCreateResponse)
def create_amazon_order_cart_task(body: AmazonOrderCartRequest) -> AmazonOrderCartTaskCreateResponse:
    # Currently the text is not parsed into fine-grained params, but this keeps
    # the interface natural-language-first so the agent brain can route here.
    payload = {
        "user_id": "demo-user",
        "text": body.text,
        "confirm": body.confirm,
    }

    task = celery_app.send_task("agents.amazon_order_cart", args=[payload])
    return AmazonOrderCartTaskCreateResponse(task_id=task.id, status="queued")


@app.get("/agents/amazon-order-cart/{task_id}", response_model=AmazonOrderCartTaskStatusResponse)
def get_amazon_order_cart_status(task_id: str) -> AmazonOrderCartTaskStatusResponse:
    async_result = AsyncResult(task_id, app=celery_app)

    if async_result.state in {"PENDING", "RECEIVED", "STARTED"}:
        return AmazonOrderCartTaskStatusResponse(
            status="running",
            message="Ordering from Amazon cart",
            result=None,
        )

    if async_result.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(async_result.result))

    if async_result.state != "SUCCESS":
        raise HTTPException(status_code=500, detail=f"Unexpected task state: {async_result.state}")

    data = async_result.result or {}

    if data.get("status") == "login_required":
        # Surface the structured login_required response directly.
        return AmazonOrderCartTaskStatusResponse(
            status="login_required",
            message=data.get("message", "Amazon login required"),
            result=None,
        )

    proof_url = _build_public_proof_url(data.get("proof_url"))

    result = AmazonOrderCartResult(
        status=data.get("status", "unknown"),
        order_id=data.get("order_id"),
        proof_url=proof_url,
        raw_result=data.get("raw_result"),
    )

    message = "Done!" if result.status == "succeeded" else "Something went wrong"

    return AmazonOrderCartTaskStatusResponse(status=result.status, message=message, result=result)


@app.post("/agents/do-anything", response_model=DoAnythingTaskCreateResponse)
def create_do_anything_task(body: DoAnythingRequest) -> DoAnythingTaskCreateResponse:
    plan = parse_do_anything_request(body.text)

    site = plan.get("site")
    steps = plan.get("steps") or []

    payload = {
        "user_id": "demo-user",
        "text": body.text,
        "site": site,
        "steps": steps,
        "confirm": body.confirm,
    }

    task = celery_app.send_task("agents.do_anything", args=[payload])
    return DoAnythingTaskCreateResponse(task_id=task.id, status="queued")


@app.get("/agents/do-anything/{task_id}", response_model=DoAnythingTaskStatusResponse)
def get_do_anything_status(task_id: str) -> DoAnythingTaskStatusResponse:
    async_result = AsyncResult(task_id, app=celery_app)

    if async_result.state in {"PENDING", "RECEIVED", "STARTED"}:
        return DoAnythingTaskStatusResponse(
            status="running",
            message="Doing anything",
            result=None,
        )

    if async_result.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(async_result.result))

    if async_result.state != "SUCCESS":
        raise HTTPException(status_code=500, detail=f"Unexpected task state: {async_result.state}")

    data = async_result.result or {}

    proof_url = _build_public_proof_url(data.get("proof_url"))

    result = DoAnythingResult(
        status=data.get("status", "unknown"),
        site=data.get("site"),
        steps=data.get("steps"),
        proof_url=proof_url,
        raw_result=data.get("raw_result", data),
    )

    status_value = result.status
    if status_value == "succeeded":
        message = "Done!"
    elif status_value == "login_required":
        message = "Login required on target site"
    elif status_value == "needs_approval":
        message = "Preview ready; needs approval"
    else:
        message = "Something went wrong"

    return DoAnythingTaskStatusResponse(status=result.status, message=message, result=result)
