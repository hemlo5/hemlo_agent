from __future__ import annotations

import redis

from app.core.config import get_settings


_settings = get_settings()
_client = redis.Redis.from_url(_settings.redis_url, decode_responses=True)


def _key(user_id: str, provider: str) -> str:
    return f"path:{user_id}:{provider}"


def mark_path_cached(user_id: str, provider: str) -> None:
    # No TTL: keep path cache as long as Redis retains data.
    _client.set(_key(user_id, provider), "1")


def is_path_cached(user_id: str, provider: str) -> bool:
    return _client.get(_key(user_id, provider)) == "1"
