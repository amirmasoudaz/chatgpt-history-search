import asyncio
import os

import aiohttp
from dotenv import find_dotenv, load_dotenv

from openai_client.calculator import TokenCalculator
from openai_client.limiter import TokenBucket
from openai_client.models import chat_models, embedding_models
from utilities.files import IOFiles


class Core:
    def __init__(self, cache_dir, chat_model: str = "gpt-3.5", embedding_model: str = "large") -> None:
        self.cache_dir = cache_dir

        self.chat_model = chat_models[chat_model]
        self.embedding_model = embedding_models[embedding_model]

        self.limiter_buckets = {
            "chat": {
                "tokens": TokenBucket(size=self.chat_model["limits"]["tpm"]),
                "requests": TokenBucket(size=self.chat_model["limits"]["rpm"])
            },
            "embedding": {
                "tokens": TokenBucket(size=self.embedding_model["limits"]["tpm"]),
                "requests": TokenBucket(size=self.embedding_model["limits"]["rpm"])
            }
        }

        self.calculator = TokenCalculator()

        _ = load_dotenv(find_dotenv("keys.env"))
        self._headers = {"Authorization": f"Bearer {os.getenv("OPENAI_API_KEY")}"}

        self._session = None
        self._req_pool = []

    async def limit(self, model: str, tokens: int, requests: int) -> None:
        await asyncio.gather(
            self.limiter_buckets[model]["tokens"].consume(tokens),
            self.limiter_buckets[model]["requests"].consume(requests)
        )

    async def make_aiohttp_session(self) -> None:
        self._session = aiohttp.ClientSession()

    async def close_aiohttp_session(self) -> None:
        await self._session.close()

    async def save_cache(self, result) -> None:
        try:
            output_path = os.path.join(self.cache_dir, f"{result['identifier']}.json")
            await IOFiles.write_json_async(output_path, result, indent=4)
        except Exception as e:
            print(f"Error: {e} on {result['identifier']}")
            output_path = os.path.join(self.cache_dir, f"{result['identifier']} - error.txt")
            await IOFiles.write_file_async(output_path, str(result))
