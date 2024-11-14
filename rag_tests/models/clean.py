from typing import Tuple
from rag_tests.models.base import VectorDBDataModel

class PostCleanedModel(VectorDBDataModel):
    entry_id: str
    platform: str = "linkedin"
    cleaned_content: str
    original_content_with_emojis: str
    author_id: str

    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "platform": self.platform,
            "author_id": self.author_id,
            "cleaned_content": self.cleaned_content,
            "original_content_with_emojis": self.original_content_with_emojis,
        }

        return self.entry_id, data
    
    