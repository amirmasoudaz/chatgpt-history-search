import json
import hashlib
import hmac
import os
import six
import time

import aiohttp
from dotenv import find_dotenv, load_dotenv
from typing import Union


class Streamer:
    _env_file = "keys.env"
    load_dotenv(find_dotenv(_env_file, usecwd=True))

    _auth_key = os.getenv("PUSHER_AUTH_KEY")
    _auth_secret = os.getenv("PUSHER_AUTH_SECRET")
    _auth_version = os.getenv("PUSHER_AUTH_VERSION")
    _app_id = os.getenv("PUSHER_APP_ID")
    _app_cluster = os.getenv("PUSHER_APP_CLUSTER")
    _app_version = os.getenv("PUSHER_APP_VERSION")

    _base = f"https://api-{_app_cluster}.pusher.com"
    _path = f"/apps/{_app_id}/events"
    _headers = {
        "X-Pusher-Library": f"pusher-http-python {_app_version}",
        "Content-Type": "application/json"
    }

    def __init__(self, session: aiohttp.ClientSession = None, channel: str = None) -> None:
        self._session = session
        self._channel = channel

    async def make_session(self):
        self._session = aiohttp.ClientSession()

    async def close_session(self):
        await self._session.close()
        self._session = None

    def _generate_query_string(self, params: dict) -> str:
        """
        Generates the query string for the Pusher event.

        :param params: Parameters to post.

        :return: Query string.
        """
        body_md5 = hashlib.md5(json.dumps(params).encode('utf-8')).hexdigest()
        query_params = {
            "auth_key": self._auth_key,
            "auth_timestamp": str(int(time.time())),
            "auth_version": self._auth_version,
            "body_md5": six.text_type(body_md5)
        }
        query_string = '&'.join(map('='.join, sorted(query_params.items(), key=lambda x: x[0])))
        auth_string = '\n'.join(["POST", self._path, query_string])
        query_params["auth_signature"] = six.text_type(hmac.new(self._auth_secret.encode('utf8'),
                                                                auth_string.encode('utf8'),
                                                                hashlib.sha256).hexdigest())
        query_string += "&auth_signature=" + query_params["auth_signature"]

        return query_string

    async def push_event(self, name: str, data: str) -> Union[dict, str]:
        """
        Triggers and pushes an event to the Pusher channel.

        :param name: Name of the event.
        :param data: Data to send with the event.

        :return: Dictionary of the response information or string.
        """
        params = {
            "name": name,
            "data": data,
            "channels": [self._channel]
        }
        query_string = self._generate_query_string(params)
        url = f"{self._base}{self._path}?{query_string}"

        try:
            async with self._session.post(
                    url=url,
                    data=json.dumps(params),
                    headers=self._headers
            ) as response:
                if response.headers.get("Content-Type") == "application/json":
                    return await response.json()
                return await response.text()
        except aiohttp.ClientError as e:
            return str(e)
