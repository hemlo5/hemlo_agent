from __future__ import annotations

from functools import lru_cache

import firebase_admin
from firebase_admin import credentials, firestore

from app.core.config import get_settings


@lru_cache
def _init_firebase() -> None:
    settings = get_settings()
    if not firebase_admin._apps:
        if settings.firebase_project_id:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {"projectId": settings.firebase_project_id})
        else:
            firebase_admin.initialize_app()


def update_task_screenshot(task_id: str, image_url: str) -> None:
    if not image_url:
        return
    _init_firebase()
    db = firestore.client()
    doc_ref = db.collection("task_screenshots").document(task_id)
    doc_ref.set({"image_url": image_url, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
