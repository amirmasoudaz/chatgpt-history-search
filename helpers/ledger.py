import os

from dotenv import load_dotenv, find_dotenv


class Ledger:
    def __init__(self):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self._paths = None
        self._configs = None

    @staticmethod
    def _get_env_variable(key, default=None, required=False, var_type=None):
        var_type = var_type or str
        value = os.environ.get(key, default)
        if required and value is None:
            raise EnvironmentError(f"Missing required environment variable: {key}")
        if value is not None and var_type != str:
            try:
                value = var_type(value)
            except ValueError:
                raise ValueError(f"Environment variable {key} must be of type {var_type.__name__}")
        return value

    @property
    def paths(self):
        if self._paths is None:
            load_dotenv(find_dotenv("config_paths.env", usecwd=True))
            dirs = {
                "exported": os.path.join(self.root, "data", self._get_env_variable("DIR_EXPORTED")),
                "processed": os.path.join(self.root, "data", self._get_env_variable("DIR_PROCESSED")),
                "vector_cache": os.path.join(self.root, "data", self._get_env_variable("DIR_VECTOR_CACHE")),
                "search_cache": os.path.join(self.root, "data", self._get_env_variable("DIR_SEARCH_CACHE"))
            }
            files = {
                "exported": os.path.join(dirs["exported"], self._get_env_variable("FILE_EXPORTED")),
                "index": os.path.join(dirs["processed"], self._get_env_variable("FILE_INDEX")),
                "msg_cache": os.path.join(dirs["processed"], self._get_env_variable("FILE_MSG_CACHE")),
                "vector_cache": os.path.join(dirs["processed"], self._get_env_variable("FILE_VECTOR_CACHE")),
                "vector_data": os.path.join(dirs["processed"], self._get_env_variable("FILE_VECTOR_DATA")),
                "msg_to_ignore": os.path.join(self.root, "data", self._get_env_variable("FILE_MSG_TO_IGNORE"))
            }
            for key, path in dirs.items():
                if key == "exported":
                    continue
                os.makedirs(path, exist_ok=True)
            self._paths = {
                "dirs": dirs,
                "files": files
            }
        return self._paths.copy()

    @property
    def configs(self):
        if self._configs is None:
            load_dotenv(find_dotenv("config_app.env", usecwd=True))
            self._configs = {
                "chat_model": self._get_env_variable("CHAT_MODEL"),
                "embedding_model": self._get_env_variable("EMBEDDING_MODEL"),
                "ignore_threshold": self._get_env_variable("IGNORE_THRESHOLD", var_type=int),
                "chunk_break_line": self._get_env_variable("CHUNK_BREAK_LINE", var_type=int),
                "chunk_trim_overlap": self._get_env_variable("CHUNK_TRIM_OVERLAP", var_type=int),
                "search_limit": self._get_env_variable("SEARCH_LIMIT", var_type=int)
            }
        return self._configs.copy()
