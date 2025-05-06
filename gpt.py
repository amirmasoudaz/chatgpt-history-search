# src/utils/gpt.py

import asyncio
import json
from pathlib import Path
import time
import uuid
from typing import Union, ClassVar, Dict

import aiofiles
from blake3 import blake3
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
import openai
import tiktoken


_ = load_dotenv(find_dotenv(), override=True)

# noinspection PyTypeChecker
class ModelSpec:
    key: ClassVar[str]
    model_name: ClassVar[str]
    category: ClassVar[str]
    context_window: ClassVar[int]
    rate_limits: ClassVar[dict]
    usage_costs: ClassVar[dict]

    max_output: ClassVar[int | None] = None
    output_dimensions: ClassVar[int | None] = None
    endpoint: ClassVar[str] = "https://api.openai.com/v1"

    _registry: ClassVar[Dict[str, "ModelSpec"]] = {}

    def __init_subclass__(cls, **kwargs):
        if getattr(cls, "key", None) in cls._registry:
            raise ValueError(f"Duplicate model key: {cls.key}")
        cls._registry[cls.key] = cls

    @classmethod
    def input_cost(cls, n_tokens: int) -> float:
        return n_tokens * cls.usage_costs["input"]

    @classmethod
    def output_cost(cls, n_tokens: int) -> float:
        if cls.category != "completions":
            return 0.0
        return n_tokens * cls.usage_costs["output"]

    @classmethod
    def get(cls, key: str) -> "ModelSpec":
        try:
            return cls._registry[key]
        except KeyError:
            raise ValueError(f"Unknown model key {key!r}") from None


class GPT4Point1(ModelSpec):
    key = "gpt-4.1"
    model_name = "gpt-4.1-2025-04-14"
    category = "completions"
    context_window = 1_047_576
    max_output = 32_768
    usage_costs = {
        "input": 0.002 / 1000,
        "output": 0.008 / 1000,
        "cached_input": 0.0005 / 1000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 5_000}


class GPT4Point1Mini(ModelSpec):
    key = "gpt-4.1-mini"
    model_name = "gpt-4.1-mini-2025-04-14"
    category = "completions"
    context_window = 1_047_576
    max_output = 32_768
    usage_costs = {
        "input": 0.0004 / 1000,
        "output": 0.0016 / 1000,
        "cached_input": 0.0001 / 1000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 5_000}


class GPT4Point1Nano(ModelSpec):
    key = "gpt-4.1-nano"
    model_name = "gpt-4.1-nano-2025-04-14"
    category = "completions"
    context_window = 1_047_576
    max_output = 32_768
    usage_costs = {
        "input": 0.0001 / 1000,
        "output": 0.0004 / 1000,
        "cached_input": 0.000025 / 1000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 5_000}


class GPT4OMini(ModelSpec):
    key = "gpt-4o-mini"
    model_name = "gpt-4o-mini"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = {
        "input": 0.00015 / 1000,
        "output": 0.00060 / 1000,
        "cached_input": 0.000075 / 1000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 5_000}


class GPT4O(ModelSpec):
    key = "gpt-4o"
    model_name = "gpt-4o-2024-11-20"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = {
        "input": 0.0025 / 1000,
        "output": 0.01 / 1000,
        "cached_input": 0.000125 / 1000,
    }
    rate_limits = {"tkn_per_min": 450_000, "req_per_min": 5_000}


class TextEmbedding3Large(ModelSpec):
    key = "text-embedding-3-large"
    model_name = "text-embedding-3-large"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 3_072
    usage_costs = {
        "input": 0.00013 / 1000
    }
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}


class TextEmbedding3Small(ModelSpec):
    key = "text-embedding-3-small"
    model_name = "text-embedding-3-small"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 1_536
    usage_costs = {
        "input": 0.00002 / 1000
    }
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}

class TextEmbeddingAda2(ModelSpec):
    key = "text-embedding-ada-002"
    model_name = "text-embedding-ada-002"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 1_536
    usage_costs = {
        "input": 0.0001 / 1000
    }
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}


class Tokenizer:
    def __init__(self, model_specs: dict = None) -> None:
        self._encoder = tiktoken.get_encoding("cl100k_base")
        if model_specs:
            self._usage_costs = model_specs["usage_costs"]

    def tokenizer(self, context: str):
        return self._encoder.encode(context)

    def count_tokens(self, context):
        try:
            if isinstance(context, str):
                return len(self.tokenizer(context))
            elif isinstance(context, list):
                if isinstance(context[0], str):
                    return len(self.tokenizer(context[0]))

                per_message = 4
                num_tokens = 0
                for message in context:
                    num_tokens += per_message
                    for key, value in message.items():
                        num_tokens += len(self.tokenizer(value))
                num_tokens += 3
                return num_tokens
            else:
                return len(self.tokenizer(str(context)))
        except Exception as e:
            print(f"Error While Counting Tokens: {e}")
            return 0

    def parse_usage(self, usage: dict) -> dict:
        if not self._usage_costs:
            raise ValueError("Usage costs not defined for the model. Pass the model_specs dictionary to the Tokenizer class")

        parsed_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        if "completion_tokens" in usage:
            parsed_usage["output_tokens"] = usage["completion_tokens"]
        if "prompt_tokens_details" in usage and usage["prompt_tokens_details"].get("cached_tokens", 0) > 0:
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            input_cost_cached = cached_tokens * self._usage_costs["cached_input"]
            input_cost_uncached = (parsed_usage["input_tokens"] - cached_tokens) * self._usage_costs["input"]
            input_cost = input_cost_cached + input_cost_uncached
            parsed_usage["input_tokens_cached"] = cached_tokens
        else:
            input_cost = parsed_usage["input_tokens"] * self._usage_costs["input"]

        parsed_usage["input_cost"] = input_cost
        if "output_tokens" in parsed_usage:
            parsed_usage["output_cost"] = parsed_usage["output_tokens"] * self._usage_costs["output"]
        parsed_usage["total_cost"] = parsed_usage["input_cost"] + parsed_usage.get("output_cost", 0)

        return parsed_usage


class TokenBucket:
    def __init__(self, size: int = 0) -> None:
        self._maximum_size = size
        self._current_size = size
        self._consume_per_second = size / 60
        self._last_fill_time = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, amount: int = 0) -> None:
        if amount == 0:
            return

        async with self._lock:
            if amount > self._maximum_size:
                raise ValueError("Amount exceeds bucket size.")

            self._refill()

            while amount > self._current_size:
                await asyncio.sleep(0.05)
                self._refill()

            self._current_size -= amount

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_fill_time
        refilled_tokens = int(elapsed * self._consume_per_second)
        self._current_size = min(self._maximum_size, self._current_size + refilled_tokens)
        self._last_fill_time = now


class Limiter:
    def __init__(self, model_specs: dict = None) -> None:
        self.tkn_limiter = TokenBucket(size=int(model_specs["rate_limits"]["tkn_per_min"] * 0.75))
        self.req_limiter = TokenBucket(size=int(model_specs["rate_limits"]["req_per_min"] * 0.95))

    def limit(self, tokens: int = 0, requests: int = 0):
        return self._LimitContextManager(self, tokens, requests)

    class _LimitContextManager:
        def __init__(self, limiter, tokens, requests):
            self.limiter = limiter
            self.tokens = tokens
            self.requests = requests
            self.output_tokens = 0

        async def __aenter__(self):
            await asyncio.gather(
                self.limiter.tkn_limiter.consume(self.tokens),
                self.limiter.req_limiter.consume(self.requests)
            )
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.output_tokens > 0:
                await self.limiter.tkn_limiter.consume(self.output_tokens)


class OpenAIClient:
    model_support = {
        "completions": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
        ],
        "embeddings": [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002"
        ]
    }

    def __init__(self, model_name: str, cache_dir: Union[str, Path, None] = None) -> None:
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.cache_backlogs = True if cache_dir else False
        if self.cache_backlogs and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.spec = ModelSpec.get(model_name)
        self._client_type = self.spec.category
        if model_name in self.model_support["completions"]:
            self._call_model = self._call_completions
            self._client_type = "completions"
        elif model_name in self.model_support["embeddings"]:
            self._call_model = self._call_embeddings
            self._client_type = "embeddings"
        else:
            raise ValueError(f"Model {model_name} not supported")

        self.model_name = self.spec.model_name
        self.tokenizer = Tokenizer(vars(self.spec))  # vars(spec) -> dict‑like view
        self.limiter = Limiter(vars(self.spec))

        self.openai = AsyncOpenAI()

        self.semaphore = asyncio.Semaphore(256)

    async def _call_completions(self, **params) -> dict:
        assert params.get("messages"), "Messages are required for completions."
        params["model"] = self.model_name

        if params.get("response_format"):
            if params["response_format"] == "text":
                params["response_format"] = {"type": "text"}
            elif params["response_format"] == "json_object":
                if "json" not in str(params["messages"]).lower():
                    raise ValueError("Context doesn't contain 'json' keyword which is required for JSON mode.")
                params["response_format"] = {"type": "json_object"}
        else:
            params["response_format"] = {"type": "text"}

        if isinstance(params["messages"], str):
            params["messages"] = [{"role": "user", "content": params["messages"]}]

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"

        async with self.limiter.limit(tokens=self.tokenizer.count_tokens(params["messages"]), requests=1) as limit_context:
            try:
                if isinstance(params["response_format"], dict):
                    response = await self.openai.chat.completions.create(**params)
                    output = response.choices[0].message.content
                    if params["response_format"]["type"] == "json_object":
                        output = json.loads(output)
                else:
                    response = await self.openai.beta.chat.completions.parse(**params)
                    output = response.choices[0].message.parsed.dict()
                usage = response.usage.to_dict()
                status, error = 200, "OK"

                usage = self.tokenizer.parse_usage(usage)
                limit_context.output_tokens = usage["output_tokens"]
            except openai.APIConnectionError as e:
                status, error = 500, e.__cause__
                print(f"API Connection Error in Completions: {error}")
            except openai.RateLimitError:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Completions: {error}")
            except openai.APIStatusError as e:
                status, error = e.status_code, e.response
                print(f"API Status Error in Completions: {error}")
            finally:
                usage = usage or {}
                return dict(params=params, output=output, usage=usage,
                            status=status, error=error)

    async def _call_embeddings(self, **params) -> dict:
        assert params.get("input"), "Input is required for embeddings."
        params["model"] = self.model_name

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"

        input_tokens = self.tokenizer.count_tokens(params["input"])
        async with self.limiter.limit(tokens=input_tokens, requests=1):
            try:
                response = await self.openai.embeddings.create(**params)
                output = [d.embedding for d in response.data]
                usage = self.tokenizer.parse_usage(response.usage.to_dict())
                status, error = 200, "OK"
            except openai.APIConnectionError as e:
                status, error = 500, e.__cause__
                print(f"API Connection Error in Embeddings: {error}")
            except openai.RateLimitError:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Embeddings: {error}")
            except openai.APIStatusError as e:
                status, error = e.status_code, e.response
                print(f"API Status Error in Embeddings: {error}")
            finally:
                usage = usage or {}
                return dict(params=params, output=output, usage=usage,
                            status=status, error=error)

    def _get_cache_path(self, identifier: str, rewrite_cache: bool) -> Path:
        cache_path = None
        if self.cache_backlogs:
            if self._client_type == "completions":
                if not rewrite_cache:
                    cache_path = self.cache_dir / f"{identifier}.json"
                else:
                    for i in range(0, 1000, 1):
                        cache_path = self.cache_dir / f"{identifier}_{i}.json"
                        if not cache_path.exists():
                            break
            else:
                cache_path = self.cache_dir / f"{identifier}.json"
        return cache_path

    async def get_response(
            self,
            identifier: str = None,
            attempts: int = 3,
            backoff: int = 1,
            body: dict = None,
            cache_response: bool = False,
            rewrite_cache: bool = False,
            **kwargs
    ) -> dict:
        if rewrite_cache and not cache_response:
            raise ValueError("Cannot rewrite cache without caching the response.")

        identifier = identifier or blake3(str(uuid.uuid4()).encode('utf-8')).hexdigest()

        cache_path = None
        if cache_response and self.cache_backlogs:
            cache_path = self._get_cache_path(identifier, rewrite_cache)
        if cache_path and cache_path.exists():
            return await self._read_json(cache_path)

        response = {}
        while attempts > 0:
            response = await self._call_model(**kwargs)
            if response["status"] < 500:
                break

            attempts -= 1
            if attempts == 0:
                return response

            await asyncio.sleep(backoff)
            backoff *= 2

        response.update({"identifier": identifier, "body": body})
        if response["body"]:
            response["body"]["completion"] = response["output"]

        if cache_response and self.cache_backlogs:
            if not isinstance(response.get("params", {}).get("response_format", {}), dict):
                response["params"]["response_format"] = {
                    "type": "json_schema",
                    "schema_name": response["params"]["response_format"].__name__
                }
            await self._write_json(cache_path, response)

        return response

    async def _run_one(self, coro):
        async with self.semaphore:
            return await coro

    async def run_batch(self, coros: list):
        wrapped = [
            asyncio.create_task(self._run_one(coro))
            for coro in coros
        ]

        results = []
        for fut in asyncio.as_completed(wrapped):
            try:
                results.append(await fut)
            except Exception as exc:
                print(f"Error in batch processing: {exc}")
                results.append({"error": str(exc), "status": 500, "output": None, "usage": None})

        return results

    async def iter_batch(self, coros: list):
        for fut in asyncio.as_completed([self._run_one(c) for c in coros]):
            yield await fut

    @staticmethod
    async def _read_json(path: str or Path, default: dict or list = None) -> dict or list:
        try:
            async with aiofiles.open(path, "r") as file:
                return json.loads(await file.read())
        except (FileNotFoundError, json.JSONDecodeError):
            default = {} if default is None else default
        return default

    @staticmethod
    async def _write_json(path: str or Path, data: dict or list, indent: int = 4, encoding: str = "utf-8") -> None:
        if isinstance(path, str):
            path = Path(path)

        if not path.suffix == ".json":
            path = path.with_suffix(".json")

        try:
            async with aiofiles.open(path, "w", encoding=encoding) as file:
                await file.write(json.dumps(data, indent=indent))
        except Exception as e:
            print(f"Error Writing JSON: {e}")


async def main():
    s = time.perf_counter()
    client = OpenAIClient(model_name="gpt-4.1-nano")
    response = await client.get_response(messages="Say '1 2 3'", max_tokens=5)
    elapsed = time.perf_counter() - s

    print(f"Elapsed Time: {elapsed:0.2f} seconds")
    print(f"Response: {response["output"]}")
    print(f"Usage: {response['usage']}")


if __name__ == "__main__":
    asyncio.run(main())
