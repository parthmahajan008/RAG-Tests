from abc import ABC,abstractmethod
from rag_tests.models.base import DataModel
from rag_tests.models.raw import PostsRawModel
from rag_tests.models.clean import PostCleanedModel
from rag_tests.utils.cleaning import clean_text
from rag_tests.core import get_logger

logger = get_logger(__name__)
class CleaningDataHandler(ABC):
    """Abstract Class for all cleaning data handlers"""
    @abstractmethod
    def clean(self, data_model: DataModel)-> DataModel:
        pass

class PostCleaningHandler(CleaningDataHandler):

    def clean(self, data_model: PostsRawModel) -> PostCleanedModel:
        cleaned_text_content = clean_text(data_model.content['text'])

        return PostCleanedModel(
            platform = data_model.platform,
            entry_id = data_model.entry_id,
            author_id= data_model.author_id, 
            cleaned_content = cleaned_text_content,
            original_content_with_emojis = data_model.content['text']
        )