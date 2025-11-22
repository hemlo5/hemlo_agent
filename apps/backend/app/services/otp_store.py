from __future__ import annotations

from typing import Optional

import redis

from app.core.config import get_settings


_settings = get_settings()
_client = redis.Redis.from_url(_settings.redis_url, decode_responses=True)


def save_otp(user_id: str, otp: str, channel: str = "sms", ttl: int = 300) -> None:
    key = f"otp:{user_id}:{channel}"
    _client.setex(key, ttl, otp)


def get_latest_otp(user_id: str, channel: str = "sms") -> Optional[str]:
    key = f"otp:{user_id}:{channel}"
    value = _client.get(key)
    return value
