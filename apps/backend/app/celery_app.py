from celery import Celery

from app.core.config import get_settings


settings = get_settings()

celery_app = Celery(
    "hemlo",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_default_queue="default",
    task_acks_late=True,
    include=["app.worker"],
)
