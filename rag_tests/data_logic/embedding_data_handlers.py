from abc import ABC, abstractmethod
from rag_tests.models.base import DataModel

from rag_tests.models.chunk import PostChunkModel
from rag_tests.models.embedded_chunk import PostEmbeddedChunkModel

from rag_tests.utils.embeddings import embedd_text

class EmbeddingDataHandler(ABC):
    @abstractmethod
    def embedd(self, data: DataModel) -> DataModel:
        pass

class PostEmbeddingDataHandler(EmbeddingDataHandler):
    def embedd(self, data_model: PostChunkModel) -> PostEmbeddedChunkModel:
        return PostEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            platform=data_model.platform,
            chunk_id=data_model.chunk_id,
            chunk_content=data_model.chunk_content,
            original_content_with_emojis=data_model.original_content_with_emojis,
            original_content_without_emojis=data_model.original_content_without_emojis,
            author_id=data_model.author_id,
            embedded_content=embedd_text(data_model.chunk_content)
        )
