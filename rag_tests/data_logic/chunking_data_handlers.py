import hashlib
from abc import ABC, abstractmethod
from rag_tests.models.base import DataModel
from rag_tests.models.clean import PostCleanedModel
from rag_tests.models.chunk import PostChunkModel
from rag_tests.utils.chunking import chunk_text
from rag_tests.core import get_logger

logger = get_logger(__name__)
class ChunkingDataHandler(ABC):
    @abstractmethod
    def chunk(self, data_model : DataModel)-> list[DataModel]:
        pass

class PostChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: DataModel) -> list[PostChunkModel]:
        data_models_list = []
        text_content = data_model.cleaned_content
        chunks = chunk_text(text_content)
        if len(chunks)>1:
            logger.info(f"Number of chunks created: {len(chunks)}")
            for chunk in chunks:
                logger.debug(chunk)
        for chunk in chunks:
            model = PostChunkModel(
                entry_id=data_model.entry_id,
                platform=data_model.platform,
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                chunk_content=chunk, 
                author_id=data_model.author_id,
                original_content_with_emojis=data_model.original_content_with_emojis,
                original_content_without_emojis=data_model.cleaned_content
            )
            data_models_list.append(model)
        return data_models_list
        