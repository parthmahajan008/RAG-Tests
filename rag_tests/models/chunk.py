from rag_tests.models.base import DataModel


class PostChunkModel(DataModel):
    entry_id: str
    platform: str
    chunk_id: str
    chunk_content: str
    original_content_with_emojis: str
    original_content_without_emojis: str
    author_id: str