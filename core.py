import os

from openai_client.client import GPTClient


class Core:
    DATA_DIR = "data"

    EXPORTED_DIR = "exported"
    PROCESSED_DIR = "processed"
    CACHE_DIR = "cache"
    LOG_DIR = "logs"

    EXPORTED_FILE = "conversations.json"
    INDEX_FILE = "index.json"
    CACHE_FILE = "cache.json"
    VECTOR_FILE = "vector.pkl"

    """
    CHAT_MODEL:
    The GPT model to use for chat continuation (turbo models)
    Options: "gpt-3.5", "gpt-4"
    """
    CHAT_MODEL = "gpt-3.5"

    """
    EMBEDDING_MODEL:
    The GPT model to use for generating embeddings
    Options: "small", "large"
    """
    EMBEDDING_MODEL = "large"

    """
    IGNORE_THRESHOLD:
    The minimum string length (characters) to index in the search cache and generate embeddings for
    """
    IGNORE_THRESHOLD = 60

    """
    CHUNK_BREAK_LINE: 
    The approximate length (tokens) of each chunk to break a message of the conversation into for indexing 
    Max: MODEL_MAX_INPUT_LENGTH - CHUNK_TRIM_OVERLAP
    """
    CHUNK_BREAK_LINE = 1024

    """
    CHUNK_TRIM_OVERLAP:
    The approximate length (tokens) of the overlap between chunks
    """
    CHUNK_TRIM_OVERLAP = 128

    """
    SEARCH_LIMIT:
    The number of search results to return
    """
    SEARCH_LIMIT = 10

    def __init__(self):
        root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(root, self.DATA_DIR)
        self.paths = {
            "dir": {
                "exported": os.path.join(data_dir, self.EXPORTED_DIR),
                "processed": os.path.join(data_dir, self.PROCESSED_DIR),
                "cache": os.path.join(data_dir, self.CACHE_DIR),
                "logs": os.path.join(data_dir, self.LOG_DIR)
            }
        }
        self.paths["file"] = {
            "exported": os.path.join(self.paths["dir"]["exported"], self.EXPORTED_FILE),
            "index": os.path.join(self.paths["dir"]["processed"], self.INDEX_FILE),
            "cache": os.path.join(self.paths["dir"]["processed"], self.CACHE_FILE),
            "vector": os.path.join(self.paths["dir"]["processed"], self.VECTOR_FILE),

        }

        for title, path in self.paths["dir"].items():
            if title != self.EXPORTED_DIR:
                os.makedirs(path, exist_ok=True)

        self.gpt_client = GPTClient(
            cache_dir=self.paths["dir"]["cache"],
            chat_model=self.CHAT_MODEL,
            embedding_model=self.EMBEDDING_MODEL
        )

        self.exported = []
        self.index = {}
        self.cache = {}
        self.vector_cache = {}
        self.search_cache = {}
        self.vector = None
