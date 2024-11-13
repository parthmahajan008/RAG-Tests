from rag_tests.core import get_logger

from rag_tests.data_logic.chunking_data_handlers import PostChunkingHandler, ChunkingDataHandler
from rag_tests.data_logic.cleaning_data_handlers import PostCleaningHandler, CleaningDataHandler
from rag_tests.data_logic.embedding_data_handlers import PostEmbeddingDataHandler, EmbeddingDataHandler

from rag_tests.models.base import DataModel
from rag_tests.models.raw import PostsRawModel

logger = get_logger(__name__)

class RawDispatcher:
    @staticmethod
    def handle_mq_message(message : dict )->DataModel:
        logger.info(f"\033[93m{message}\033[0m")
        entry_id = message["id"]
        if entry_id not in message:
            message["entry_id"] = message.pop("id")
        model_obj =  PostsRawModel(**message)
        logger.info("Recieved message", extra={"data_type": "linkedin_posts", "model_obj": model_obj})
        logger.info(f"\033[93m{model_obj}\033[0m")
        logger.info("Message converted to model object",extra={"data_type": "linkedin_posts", "model_obj": model_obj})
        return model_obj

class CleaningHandlerFactory:
    @staticmethod
    def create_handler() -> CleaningDataHandler:
        return PostCleaningHandler()
    
class CleaningDispatcher:
    cleaning_factory  = CleaningHandlerFactory()
    @classmethod
    def dispatch_cleaner(cls, data_model :DataModel)->DataModel:
        logger.info(data_model)
        handler = cls.cleaning_factory.create_handler()
        clean_model = handler.clean(data_model)

        logger.info(
            "Data cleaned successfully.",
            extra={"data_type": "LinkedIn Posts", "cleaned_content": clean_model.cleaned_content},
        )

        return clean_model
    
class ChunkingHandlerFactory:
    @staticmethod
    def create_handler() -> ChunkingDataHandler:
        return PostChunkingHandler()
    
class ChunkingDispatcher:
    chunking_factory  = ChunkingHandlerFactory()
    @classmethod
    def dispatch_chunker(cls, data_model :DataModel)->DataModel:
        handler = cls.chunking_factory.create_handler()
        chunk_models = handler.chunk(data_model)

        logger.info(
            "Cleaned content chunked successfully.",
            extra={"data_type": "Linkedin Cleaned Posts", "chunked_content": chunk_models},
        )

        return chunk_models
    
class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler() -> EmbeddingDataHandler:
        return PostEmbeddingDataHandler()
    
class EmbeddingDispatcher:
    embedding_factory  = EmbeddingHandlerFactory()
    @classmethod
    def dispatch_embedder(cls, data_model :DataModel)->DataModel:
        handler = cls.embedding_factory.create_handler()
        embedded_chunk_model= handler.embedd(data_model)
        
        logger.info(
            "Chunk embedded successfully.",
            extra={
                "data_type": "Linkedin Posts Embedded Chunks",
                "embedding_len": len(embedded_chunk_model.embedded_content),
            },
        )

        return embedded_chunk_model