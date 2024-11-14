import logging
from datetime import datetime
import os



import logging
from datetime import datetime
import os

LOG_DIR = "rag_tests/logs"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
CURRENT_LOG_FILE = f"log_{CURRENT_TIME_STAMP}.log"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, CURRENT_LOG_FILE)

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'collection_name'):
            record.collection_name = 'N/A'
        return super().format(record)

formatter = CustomFormatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s - %(collection_name)s')

handler = logging.FileHandler(LOG_FILE_PATH, mode='w')
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f'{msg}', {**kwargs, 'extra': {**self.extra, **kwargs.get('extra', {})}}

def get_logger(cls: str):
    logger = logging.getLogger(cls)
    return CustomAdapter(logger, {})

# Example usage
logger = get_logger(__name__)
logger.info("Successfully accessed the collection", extra={'collection_name': 'cleaned_posts'})


