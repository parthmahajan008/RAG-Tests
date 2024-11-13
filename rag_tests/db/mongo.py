from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from rag_tests.config import settings

from rag_tests.core import get_logger

logger = get_logger(__name__)
# logger = get_logger("MongoDatabaseConnector")

class MongoDatabaseConnector:
    """Singleton class to connect to MongoDB database."""

    _instance: MongoClient = None 

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.MONGO_DATABASE_HOST)
            except ConnectionFailure as e:
                logger.info(f"Couldn't connect to the database: {str(e)}")
                raise

        logger.info(
            f"Connection to database with uri: {settings.MONGO_DATABASE_HOST} successful"
        )
        return cls._instance

    def get_database(self):
        return self._instance[settings.MONGO_DATABASE_NAME]

    def close(self):
        if self._instance:
            self._instance.close()
            logger.info("Connected to database has been closed.")


connection = MongoDatabaseConnector()
