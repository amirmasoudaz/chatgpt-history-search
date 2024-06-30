import os

from openai_client.client import OpenAINative

from utilities.files import FileTools


class Core:
    DATA_DIR = "data"

    EXPORTED_DIR = "exported"
    PROCESSED_DIR = "processed"
    VECTOR_CACHE_DIR = "vector_cache"
    SEARCH_CACHE_DIR = "search_cache"

    EXPORTED_FILE = "conversations.json"
    INDEX_FILE = "index.json"
    MSG_CACHE_FILE = "msg_cache.json"
    VECTOR_CACHE_FILE = "vector_cache.json"
    VECTOR_DATA_FILE = "vector_data.pkl"
    DIGESTION_INFO_FILE = "digestion_info.json"

    """
    CHAT_MODEL:
    The GPT model to use for chat continuation (turbo models)
    Options: "gpt-3.5", "gpt-4o", "gpt-4"
    """
    CHAT_MODEL = "gpt-4o"

    """
    TEMPERATURE:
    The temperature to use for chat continuation
    Range: 0.0 - 2.0
    """
    CHAT_TEMPERATURE = 0.0

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
                "vector_cache": os.path.join(data_dir, self.VECTOR_CACHE_DIR),
                "search_cache": os.path.join(data_dir, self.SEARCH_CACHE_DIR)
            }
        }
        self.paths["file"] = {
            "exported": os.path.join(self.paths["dir"]["exported"], self.EXPORTED_FILE),
            "index": os.path.join(self.paths["dir"]["processed"], self.INDEX_FILE),
            "msg_cache": os.path.join(self.paths["dir"]["processed"], self.MSG_CACHE_FILE),
            "vector_cache": os.path.join(self.paths["dir"]["processed"], self.VECTOR_CACHE_FILE),
            "vector_data": os.path.join(self.paths["dir"]["processed"], self.VECTOR_DATA_FILE),
            "digestion_info": os.path.join(data_dir, self.DIGESTION_INFO_FILE)
        }

        for title, path in self.paths["dir"].items():
            if title != self.EXPORTED_DIR:
                os.makedirs(path, exist_ok=True)

        self.gpt_client = OpenAINative(
            cache_dir=self.paths["dir"]["vector_cache"],
            chat_model=self.CHAT_MODEL,
            embedding_model=self.EMBEDDING_MODEL
        )
        self.file_tools = FileTools()

        self.digestion_info = {
            "to_ignore": [],
        }
        self.indexed_data = {}
        self.vector_cache = {}
        self.search_cache = {}
        self.vector_data = None
