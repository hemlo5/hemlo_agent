from typing import Optional, List

from pydantic import BaseModel


class OtpSubmitRequest(BaseModel):
    user_id: str
    otp: str
    channel: Optional[str] = "sms"
    source: Optional[str] = None


class OtpSubmitResponse(BaseModel):
    status: str


class StripeRefundRequest(BaseModel):
    text: str
    charge_id: Optional[str] = None
    amount_cents: Optional[int] = None
    currency: Optional[str] = None
    reason: Optional[str] = None
    use_api: bool = False
    confirm: bool = False


class StripeRefundTaskCreateResponse(BaseModel):
    task_id: str
    status: str


class StripeRefundResult(BaseModel):
    status: str
    charge_id: Optional[str] = None
    refund_id: Optional[str] = None
    amount_cents: Optional[int] = None
    currency: Optional[str] = None
    reason: Optional[str] = None
    proof_url: Optional[str] = None
    raw_result: Optional[dict] = None


class StripeRefundTaskStatusResponse(BaseModel):
    status: str
    message: str
    result: Optional[StripeRefundResult] = None


class AmazonOrderCartRequest(BaseModel):
    text: str
    confirm: bool = False


class AmazonOrderCartTaskCreateResponse(BaseModel):
    task_id: str
    status: str


class AmazonOrderCartResult(BaseModel):
    status: str
    order_id: Optional[str] = None
    proof_url: Optional[str] = None
    raw_result: Optional[dict] = None


class AmazonOrderCartTaskStatusResponse(BaseModel):
    status: str
    message: str
    result: Optional[AmazonOrderCartResult] = None


class DoAnythingRequest(BaseModel):
    text: str
    confirm: bool = False


class DoAnythingTaskCreateResponse(BaseModel):
    task_id: str
    status: str


class DoAnythingResult(BaseModel):
    status: str
    site: Optional[str] = None
    steps: Optional[List[str]] = None
    proof_url: Optional[str] = None
    raw_result: Optional[dict] = None


class DoAnythingTaskStatusResponse(BaseModel):
    status: str
    message: str
    result: Optional[DoAnythingResult] = None
