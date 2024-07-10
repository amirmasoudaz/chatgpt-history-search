import asyncio
import json
import os
import time
import uuid

import aiofiles
import aiohttp

from gpt.completions import OpenAICompletions
from gpt.embeddings import OpenAIEmbeddings


class OpenAI:
    def __init__(self, model_name: str, backlogs_dir: str or None = None) -> None:
        """
        Initializes the OpenAI client

        :param model_name: name of the model to use
        :param backlogs_dir: directory to save the backlogs

        :return: None
        """
        self.model_name = model_name

        if self.model_name in OpenAICompletions.model_support:
            self.client = OpenAICompletions(self.model_name)
        elif self.model_name in OpenAIEmbeddings.model_support:
            self.client = OpenAIEmbeddings(self.model_name)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

        self.save_backlogs = True if backlogs_dir else False
        self.backlogs_dir = backlogs_dir

        self._session = None
        self._pool = []

    def __repr__(self):
        return f"OpenAI(model_name={self.model_name})"

    async def make_session(self):
        self._session = aiohttp.ClientSession()

    async def close_session(self):
        await self._session.close()
        self._session = None

    @staticmethod
    async def _read_json(path: str, default: dict or list = None) -> dict or list:
        """
        Read a JSON file asynchronously

        :param path: Path to the JSON file
        :param default: Default value to return if file is not found or corrupted

        :return: JSON data
        """
        try:
            async with aiofiles.open(path, "r") as file:
                return json.loads(await file.read())
        except (FileNotFoundError, json.JSONDecodeError):
            return {} if default is None else default

    @staticmethod
    async def _write_json(path: str, data: dict, indent: int = 4) -> None:
        """
        Write data to a JSON file asynchronously

        :param path: Path to the JSON file
        :param data: Data to write
        :param indent: Indentation level

        :return: None
        """
        if not path.endswith(".json"):
            path += ".json"

        try:
            async with aiofiles.open(path, "w") as file:
                await file.write(json.dumps(data, indent=indent))
        except Exception as e:
            print(f"Error Writing JSON: {e}")

    async def _save_resp(self, result) -> None:
        """
        Save the result to the cache directory

        :param result: dictionary of the response information

        :return: None
        """
        if not self.save_backlogs:
            return

        path = os.path.join(self.backlogs_dir, f"{result['identifier']}.json")
        cache = await self._read_json(path, default=[]) + [result]
        await self._write_json(path, cache, indent=4)

    async def get_response(
            self,
            context: list or str,
            identifier: str = None,
            max_attempts: int = 3,
            backoff_time: int = 1,
            session: aiohttp.ClientSession or None = None,
            body: dict = None,
            **kwargs
    ) -> dict:
        """
        Get the response from the OpenAI model

        :param context: context to post to the model
        :param identifier: unique identifier for the request
        :param max_attempts: maximum number of attempts
        :param backoff_time: backoff time between attempts
        :param session: aiohttp session for batch processing
        :param body: body for the request to track the request
        :param kwargs: additional parameters to pass to the model

        :return: dictionary of the response information
        """
        init_ts = time.time()
        identifier = identifier if identifier else str(uuid.uuid4())

        if self._session:
            response = await self.client.call_model(context, self._session, **kwargs)
        elif session:
            response = await self.client.call_model(context, session, **kwargs)
        else:
            async with aiohttp.ClientSession() as disp_session:
                response = await self.client.call_model(context, disp_session, **kwargs)

        last_ts = time.time()
        response.update({
            "init_ts": init_ts,
            "identifier": identifier,
            "last_ts": last_ts,
            "duration": last_ts - init_ts,
            "body": body
        })
        await self._save_resp(response)

        if response["status"] >= 500:
            if response["status"] != 503 and max_attempts < 0:
                return response

            await asyncio.sleep(backoff_time)
            return await self.get_response(
                context=context,
                identifier=identifier,
                max_attempts=max_attempts - 1,
                backoff_time=backoff_time * 2,
                session=session,
                body=body,
                **kwargs)

        if response["body"]:
            response["body"][self.client.model_type.title()[:-1]] = response["output"]

        return response

    def add_get_response(
            self,
            context: list or str,
            identifier: str = None,
            max_attempts: int = 3,
            backoff_time: int = 1,
            body: dict = None,
            **kwargs
    ) -> None:
        """
        Add a request to the requests pool for batch processing


        :param context: context to post to the model
        :param identifier: unique identifier for the request
        :param max_attempts: maximum number of attempts
        :param backoff_time: backoff time between attempts
        :param body: body for the request to track the request
        :param kwargs: additional parameters to pass to the model

        :return: None
        """
        self._pool.append((context, identifier, max_attempts, backoff_time, body, kwargs))

    async def batch_get_response(self):
        """
        Batch process all the requests in the requests pool

        :return: list of dictionaries of the response information
        """
        if not self._pool:
            return []

        await self.make_session()
        tasks_pool = [
            self.get_response(
                context=context,
                identifier=identifier,
                max_attempts=max_attempts,
                backoff_time=backoff_time,
                body=body,
                **kwargs)
            for context, identifier, max_attempts, backoff_time, body, kwargs in self._pool
        ]
        responses = await asyncio.gather(*tasks_pool)

        await self.close_session()
        self._pool = []

        return responses
