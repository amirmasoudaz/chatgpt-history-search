import json
from typing import Literal, Union

import aiohttp
from dotenv import find_dotenv, get_key
from openai import AsyncOpenAI

from gpt.limiter import Limiter
from gpt.streamer import Streamer
from gpt.tokenizer import Tokenizer
from gpt.models import models


class OpenAICompletions:
    _env_file = "keys.env"
    _env_key = "API_KEY_OPENAI"
    _api_key = get_key(find_dotenv(_env_file, usecwd=True), _env_key)

    model_support = [
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-4o"
    ]
    model_type = "completions"

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the OpenAI Chat native client

        :param model_name: name of the model to use

        :return: None
        """
        self.specs = models[model_name]
        self._name = self.specs["model_name"]
        self._endpoint = self.specs["endpoint"]

        self._openai = AsyncOpenAI(api_key=self._api_key)
        self._limiter = Limiter(self.specs)
        self.tokenizer = Tokenizer(self.specs)

    async def _stream(self, session: aiohttp.ClientSession, params: dict, channel: str) -> dict:
        """
        Streams the response from the OpenAI model to the Pusher channel

        :param params: parameters used to generate the response
        :param channel: Pusher channel to stream the response

        :return: dictionary of the response information
        """
        output, usage, status, error = "", {}, 500, "INCOMPLETE"

        streamer = Streamer(session=session, channel=channel)
        listener = await self._openai.chat.completions.create(
            model=params["model"],
            messages=params["messages"],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            response_format={"type": params["response_format"]["type"]},
            stream_options={"include_usage": True},
            stream=True)

        await streamer.push_event('new-response', "")
        async for chunk in listener:
            if not chunk.choices and chunk.usage:
                usage = dict(chunk.usage)
                error, status = "OK", 200
                break
            token = chunk.choices[0].delta.content
            if token is not None:
                await streamer.push_event('new-token', token)
                output += token
        await streamer.push_event('response-finished', error)

        return {
            "output": output,
            "usage": usage,
            "status": status,
            "error": error,
            "data": {}
        }

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
            context: Union[str, list],
            session: aiohttp.ClientSession,
            max_tokens: Union[int, None] = None,
            temperature: float = 0.0,
            response_format: Literal["text", "json_object"] = "text",
            stream: Union[bool, None] = False,
            channel: str = ""
    ) -> dict:
        """
        Posts the context to the OpenAI Chat model

        :param context: context to post
        :param session: aiohttp session
        :param max_tokens: maximum tokens to generate
        :param temperature: temperature for the model
        :param response_format: "text" or "json_object"
        :param stream: whether to stream the response
        :param channel: pusher channel to stream the response

        :return: dictionary of the response information
        """
        if response_format == "json_object" and "json" not in str(context).lower():
            raise ValueError("Context doesn't contain 'json' keyword which is required for JSON mode.")
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]

        params = {
            "model": self._name,
            "messages": context,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": response_format},
            "stream": stream,
        }

        await self._limiter.limit(tokens=self.tokenizer.count_tokens(context), requests=1)
        if stream and channel:
            response = await self._stream(session, params, channel)
        else:
            response = await self._post(session, params)

        if response["status"] == 200:
            if stream:
                output = response.pop("output")
                usage = response.pop("usage")
            else:
                output = response["data"]["choices"][0]["message"].pop("content", None)
                usage = response["data"].pop("usage", {})
            if response_format == "json_object":
                output = json.loads(output)
            usage = self.tokenizer.parse_usage(usage)
            await self._limiter.limit(tokens=usage["output_tokens"])
            response.update({
                "params": params,
                "output": output,
                "usage": usage
            })

        return response
