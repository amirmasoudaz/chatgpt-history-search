import asyncio
import time


class TokenBucket:
    def __init__(self, size):
        self._maximum_size = size
        self._current_size = size
        self._consume_per_second = size / 60
        self._last_fill_time = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, amount):
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

    def _refill(self):
        now = time.time()
        elapsed = now - self._last_fill_time
        refilled_tokens = elapsed * self._consume_per_second
        self._current_size = min(self._maximum_size, self._current_size + refilled_tokens)
        self._last_fill_time = now
