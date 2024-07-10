import asyncio
import time


class TokenBucket:
    def __init__(self, size: int = 0) -> None:
        """
        Initializes the Token Bucket Rate Limiter

        :param size: size of the bucket

        :return: None
        """

        self._maximum_size = size
        self._current_size = size
        self._consume_per_second = size / 60
        self._last_fill_time = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, amount: int = 0) -> None:
        """
        Consumes the amount of tokens

        :param amount: number of tokens to consume

        :return: None
        """

        if amount == 0:
            return

        async with self._lock:
            if amount > self._maximum_size:
                raise ValueError("Amount exceeds bucket size.")

            self._refill()

            while amount > self._current_size:
                await asyncio.sleep(0.1)
                self._refill()

            self._current_size -= amount

    def _refill(self) -> None:
        """
        Refills the bucket with tokens

        :return: None
        """

        now = time.time()
        elapsed = now - self._last_fill_time
        refilled_tokens = int(elapsed * self._consume_per_second)
        self._current_size = min(self._maximum_size, self._current_size + refilled_tokens)
        self._last_fill_time = now


class Limiter:
    def __init__(self, model_specs: dict = None) -> None:
        """
        Initializes the OpenAI API Rate Limiter

        :param model_specs: dictionary of model specifications

        :return: None
        """

        self._tkn_limiter = TokenBucket(size=model_specs["rate_limits"]["tkn_per_min"] * 0.9)
        self._req_limiter = TokenBucket(size=model_specs["rate_limits"]["req_per_min"] * 0.9)

    async def limit(self, tokens: int = 0, requests: int = 0) -> None:
        """
        Limits the number of tokens and requests

        :param tokens: number of tokens to consume
        :param requests: number of requests to consume

        :return: None
        """

        await asyncio.gather(
            self._tkn_limiter.consume(tokens),
            self._req_limiter.consume(requests)
        )
