from typing import Optional
from rag_tests.models.base import DataModel


class PostsRawModel(DataModel):
    platform: str
    content: dict
    author_id: str | None = None

