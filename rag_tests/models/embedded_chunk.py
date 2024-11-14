from typing import Tuple
from rag_tests.models.base import VectorDBDataModel
import numpy as np 
# numpy was added with langchain 0.3.3 (or higher)

class PostEmbeddedChunkModel(VectorDBDataModel):
    entry_id: str
    platform: str = "linkedin"
    author_id: str | None = None
    chunk_id: str
    chunk_content: str
    original_content_with_emojis: str
    original_content_without_emojis: str
    embedded_content: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id":self.entry_id,
            "platform":self.platform,
            "owner_id":self.author_id,
            "content":self.chunk_content,
            "original_content_with_emojis":self.original_content_with_emojis,
            "original_content_without_emojis":self.original_content_without_emojis,
        }
    
        return self.chunk_id, self.embedded_content, data