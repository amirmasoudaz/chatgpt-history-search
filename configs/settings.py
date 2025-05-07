# config/settings.py

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    chat_model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-large"

    ignore_threshold: int = 60
    chunk_size: int = 1024
    chunk_overlap: int = 128
    search_limit: int = 10

    data_root: Path = Path("data")

    class Config:
        env_file = ".env"
        frozen = True

    class Paths:
        root = Path(__file__).resolve().parent.parent
        data_path = root / "data"

        chat_cache_dir = data_path / "cache/chat"
        embed_cache_dir = data_path / "cache/embed"
        search_cache_dir = data_path / "cache/search"

        exported_file = data_path / "exported/conversations.json"
        index_file = data_path / "processed/index.json"
        msg_cache_file = data_path / "cache/msg_cache.json"
        vector_cache_file = data_path / "cache/vector_cache.json"
        vector_data_file = data_path / "cache/vector_data.pkl"
        msg_to_ignore_file = data_path / "cache/msg_to_ignore.json"

