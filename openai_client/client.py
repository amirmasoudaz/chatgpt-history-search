import asyncio
import base64
import json
from typing import List, Dict
import uuid

import aiohttp
import numpy as np

from openai_client.core import Core


class OpenAINative(Core):
    def __init__(
            self,
            cache_dir: str,
            chat_model: str = "gpt-4",
            embedding_model: str = "large"
    ) -> None:
        super().__init__(
            cache_dir=cache_dir,
            chat_model=chat_model,
            embedding_model=embedding_model
        )

    async def get_embedding(
            self,
            context: str,
            identifier: str = "",
            attempts: int = 3,
            backoff_time: float = 0.5
    ) -> Dict[str, str]:
        identifier = str(uuid.uuid4()) if not identifier else identifier
        input_tokens = self.calculator.count_tokens(context)
        await self.limit(
            model="embedding",
            tokens=input_tokens,
            requests=1)

        params = {
            "model": self.embedding_model["name"],
            "input": [context],
            "encoding_format": "base64"
        }

        async with self._session.post(
                url=self.embedding_model["endpoint"],
                headers=self._headers,
                json=params
        ) as response:
            status = response.status
            content = await response.json()

            if status == 200:
                output = np.frombuffer(base64.b64decode(content["data"][0]["embedding"]), dtype="float32").tolist()
                usage = self.calculator.calc_usage(content["usage"], self.embedding_model)
            elif (500 <= status or status == 429) and attempts > 0:
                await asyncio.sleep(backoff_time)
                return await self.get_embedding(
                    context=context,
                    identifier=identifier,
                    attempts=attempts - 1,
                    backoff_time=backoff_time * 2
                )
            else:
                output, usage = None, None

        result = {
            "identifier": identifier,
            "params": params,
            "output": output,
            "usage": usage,
            "status": status,
            "response": content
        }
        await self.save_cache(result)

        return result

    async def get_completion(
            self,
            context: List[Dict[str, str]],
            identifier: str = "",
            temperature: float = 0.0,
            max_tokens: int or None = None,
            response_format: str = "text",  # or "json_object"
            attempts: int = 3,
            backoff_time: float = 0.5
    ) -> Dict[str, str]:
        identifier = str(uuid.uuid4()) if not identifier else identifier

        input_tokens = self.calculator.count_tokens(context)
        await self.limit(model="chat", tokens=input_tokens, requests=1)

        params = {
            "model": self.chat_model["name"],
            "messages": context,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": response_format}
        }

        async with self._session.post(
                url=self.chat_model["endpoint"],
                headers=self._headers,
                json=params
        ) as response:
            status = response.status
            content = await response.json()

            if status == 200:
                output = content["choices"][0]["message"]["content"]
                if response_format == "json_object" and isinstance(output, str):
                    output = json.loads(output)
                usage = self.calculator.calc_usage(content["usage"], self.chat_model)
                await self.limit(model="chat", tokens=usage["output_tokens"], requests=0)
            elif 500 <= status and attempts > 0:
                await asyncio.sleep(backoff_time * (1 + np.random.uniform(0, 0.2)))
                return await self.get_completion(
                    context=context,
                    identifier=identifier,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    attempts=attempts - 1,
                    backoff_time=backoff_time * 2
                )
            else:
                output, usage = None, None

        result = {
            "identifier": identifier,
            "params": params,
            "output": output,
            "usage": usage,
            "status": status,
            "response": content
        }
        # await self.save_cache(result)

        return result

    async def trigger_requests(self):
        if not self._req_pool:
            print("No requests to trigger!")
            return []

        async with aiohttp.ClientSession() as self._session:
            request_responses = await asyncio.gather(*self._req_pool)
            responses = [response for response in request_responses]

        self._req_pool = []

        return responses

    def add_request(self, context, identifier, engine, **kwargs) -> None:
        if engine == "embedding":
            self._req_pool.append(self.get_embedding(context=context, identifier=identifier, **kwargs))
        elif engine == "chat":
            self._req_pool.append(self.get_completion(context=context, identifier=identifier, **kwargs))
        else:
            raise ValueError(f"Invalid engine type: {engine}")

    async def call_model(self, context, identifier, engine, **kwargs) -> dict:
        async with aiohttp.ClientSession() as self._session:
            if engine == "embedding":
                return await self.get_embedding(context=context, identifier=identifier, **kwargs)
            elif engine == "chat":
                return await self.get_completion(context=context, identifier=identifier, **kwargs)
            else:
                print(f"Invalid engine type: {engine}")
                return {}
