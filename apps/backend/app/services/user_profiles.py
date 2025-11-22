from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings
from app.schemas import ElectricityProvider


def _get_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def _ensure_collection(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if "user_profiles" not in names:
        client.create_collection(
            collection_name="user_profiles",
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )


def _derive_key() -> bytes:
    settings = get_settings()
    return hashlib.sha256(settings.secret_key.encode("utf-8")).digest()


def _encrypt(value: str) -> str:
    """Simple symmetric encryption using XOR with a key-derived keystream and base64."""

    key = _derive_key()
    nonce = os.urandom(16)
    data = value.encode("utf-8")
    keystream = hashlib.sha256(key + nonce).digest()
    cipher_bytes = bytes(b ^ keystream[i % len(keystream)] for i, b in enumerate(data))
    payload = nonce + cipher_bytes
    return base64.urlsafe_b64encode(payload).decode("ascii")


def _decrypt(value: str) -> Optional[str]:
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii"))
    except Exception:
        return None
    if len(raw) < 17:
        return None
    nonce = raw[:16]
    cipher_bytes = raw[16:]
    key = _derive_key()
    keystream = hashlib.sha256(key + nonce).digest()
    data = bytes(b ^ keystream[i % len(keystream)] for i, b in enumerate(cipher_bytes))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def upsert_user_profile(
    user_id: str,
    provider: str,
    phone_number: Optional[str],
    email: Optional[str],
    consumer_id: Optional[str],
) -> None:
    client = _get_client()
    _ensure_collection(client)

    payload: Dict[str, Any] = {
        "user_id": user_id,
        "provider": provider,
    }
    if phone_number:
        payload["phone_number"] = _encrypt(phone_number)
    if email:
        payload["email"] = _encrypt(email)
    if consumer_id:
        payload["consumer_id"] = _encrypt(consumer_id)

    point = PointStruct(id=user_id, vector=[0.0], payload=payload)
    client.upsert(collection_name="user_profiles", points=[point])


def get_user_profile_for_provider(
    user_id: str,
    provider: ElectricityProvider,
) -> Optional[Dict[str, str]]:
    client = _get_client()
    _ensure_collection(client)

    result = client.retrieve(collection_name="user_profiles", ids=[user_id])
    if not result:
        return None

    payload = result[0].payload or {}
    if payload.get("provider") != provider.value:
        return None

    profile: Dict[str, str] = {}

    enc_phone = payload.get("phone_number")
    if isinstance(enc_phone, str):
        phone = _decrypt(enc_phone)
        if phone:
            profile["phone_number"] = phone

    enc_email = payload.get("email")
    if isinstance(enc_email, str):
        email = _decrypt(enc_email)
        if email:
            profile["email"] = email

    enc_consumer_id = payload.get("consumer_id")
    if isinstance(enc_consumer_id, str):
        consumer_id = _decrypt(enc_consumer_id)
        if consumer_id:
            profile["consumer_id"] = consumer_id

    return profile or None
