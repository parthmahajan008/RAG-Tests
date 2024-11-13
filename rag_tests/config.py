from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# okay we have to only connect to mongodb to retrieve the documents, and then we shall process in a loop for now
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    MONGO_DATABASE_HOST: str = Field(..., env="MONGO_DATABASE_HOST")
    MONGO_DATABASE_NAME: str = Field(..., env="MONGO_DATABASE_NAME")

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 256
    EMBEDDING_SIZE: int = 384 #embedding size support by the model
    EMBEDDING_MODEL_DEVICE: str = "cpu"

    # Qdrant config
    QDRANT_HOST: str = Field(..., env="QDRANT_HOST")
    QDRANT_PORT: int = Field(..., env="QDRANT_PORT")

    OPENAI_MODEL_ID: str = "gpt-4"
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

settings = Settings()
