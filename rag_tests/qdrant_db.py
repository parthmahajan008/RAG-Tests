from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


from qdrant_client.http.models import Distance, Batch, VectorParams

from rag_tests.config import settings
from rag_tests.core import get_logger

logger = get_logger(__name__)

class QdrantDatabaseConnector:
    _instance: QdrantClient |None = None
    
    def __init__(self) ->None:
        if self._instance is None:
            try :
                self._instance = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )
            except UnexpectedResponse as e:
                logger.info(f"\033[91m{e}\033[0m")
        else:
            logger.info("\033[92mUsing existing Qdrant client instance\033[0m")

    def get_collection(self, collection_name: str):
        return self._instance.get_collection(collection_name=collection_name)

    def create_non_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name, vectors_config={}
        )

    def create_vector_collection(self, collection_name: str):
        if self._instance.get_collection(collection_name=collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_SIZE, distance=Distance.COSINE
            ),
        )

    def write_data(self, collection_name: str, points: Batch):
        try:
            self._instance.upsert(collection_name=collection_name, points=points)
        except Exception:
            logger.info("An error occurred while inserting data to QDRANT DB")

            raise

    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: models.Filter | None = None,
        limit: int = 3,
    ) -> list:
        return self._instance.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def scroll(self, collection_name: str, limit: int):
        return self._instance.scroll(collection_name=collection_name, limit=limit)

    def close(self):
        if self._instance:
            self._instance.close()

            logger.info("Connected to database has been closed.")