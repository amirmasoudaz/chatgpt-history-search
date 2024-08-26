import base64
from typing import Literal

import aiohttp
from dotenv import find_dotenv, get_key
import numpy as np

from gpt.limiter import Limiter
from gpt.tokenizer import Tokenizer
from gpt.models import models


class OpenAIEmbeddings:
    _env_file = "keys.env"
    _env_key = "API_KEY_OPENAI"
    _api_key = get_key(find_dotenv(_env_file, usecwd=True), _env_key)

    model_support = [
        "text-embedding-3-large",
        "text-embedding-3-small"
    ]
    model_type = "embeddings"

    def __init__(self, model_name: str = "text-embedding-3-large") -> None:
        """
        Initializes the OpenAI Embedding native client

        :param model_name: name of the model to use

        :return: None
        """
        self.specs = models[model_name]
        self._name = self.specs["model_name"]
        self._endpoint = self.specs["endpoint"]

        self._limiter = Limiter(self.specs)
        self.tokenizer = Tokenizer(self.specs)

    async def _post(self, session: aiohttp.ClientSession, params: dict) -> dict:
        """
        Posts the params to the OpenAI model API

        :param params: parameters used to generate the response

        :return: dictionary of the response information
        """
        async with session.post(
                headers={
                    "Authorization": f"Bearer {self._api_key}"
                },
                url=self._endpoint,
                json=params
        ) as resp:
            data = await resp.json()
            status = resp.status
            error = resp.reason

        return {
            "params": params,
            "status": status,
            "error": error,
            "data": data
        }

    async def call_model(
            self,
            context: str,
            session: aiohttp.ClientSession,
            encoding_format: Literal["float", "base64"] = "float"
    ) -> dict:
        """
        Posts the context to the OpenAI Embedding model

        :param context: context to post
        :param session: aiohttp session
        :param encoding_format: "float" or "base64"

        :return: dictionary of the response information
        """
        params = {
            "model": self._name,
            "encoding_format": encoding_format,
            "input": [context],
        }

        await self._limiter.limit(tokens=self.tokenizer.count_tokens(context), requests=1)
        response = await self._post(session, params)

        if response["status"] == 200:
            output = response["data"]["data"][0].pop("embedding", None)
            usage = response["data"].pop("usage", {})
            if encoding_format == "base64":
                output = np.frombuffer(base64.b64decode(output), dtype="float32").tolist()
            usage = self.tokenizer.parse_usage(usage)
            response.update({
                "params": params,
                "output": output,
                "usage": usage
            })

        return response
