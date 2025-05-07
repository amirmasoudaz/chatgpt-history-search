# src/utils/files.py

import asyncio
import json
from pathlib import Path

import aiofiles


class AsyncFiles:
    concurrent_operations = 300
    _semaphore = asyncio.Semaphore(concurrent_operations)

    @staticmethod
    async def read_files(path: str or Path, dtype: str = "json", default=None) -> dict:
        tasks, idents = [], []
        if isinstance(path, str):
            path = Path(path)

        if path.exists() and path.is_dir():
            for file in path.iterdir():
                if file.suffix == f".{dtype}":
                    idents.append(file.stem if dtype != "all" else file.name)

                    if dtype == "json":
                        tasks.append(AsyncFiles.read_json(file, default=default))
                    else:
                        tasks.append(AsyncFiles.read_file(file, default=default))

        results = await asyncio.gather(*tasks)
        contents = {ident: content for ident, content in zip(idents, results)}

        return contents

    @staticmethod
    async def write_files(path: str or Path, contents: dict, dtype: str, encoding: str = "utf-8") -> None:
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        tasks = []
        for ident, content in contents.items():
            file_path = path / f"{ident}.{dtype}"
            if dtype == "json":
                tasks.append(AsyncFiles.write_json(file_path, content, encoding=encoding))
            else:
                tasks.append(AsyncFiles.write_file(file_path, content, encoding=encoding))

        await asyncio.gather(*tasks)

    @staticmethod
    async def read_file(path: str or Path, default=None, route=False, encoding=None) -> any:
        if route:
            if isinstance(path, str):
                path = Path(path)

            dtype = path.suffix[1:]
            if dtype == "json":
                return await AsyncFiles.read_json(path, default=default)
            else:
                return await AsyncFiles.read_file(path, default=default, route=False, encoding=encoding)
        async with AsyncFiles._semaphore:
            try:
                async with aiofiles.open(path, "r", encoding=encoding) as file:
                    return await file.read()
            except Exception as e:
                print(f"Error Reading File: {e}")

        return default

    @staticmethod
    async def write_file(path: str or Path, data: str, encoding: str = "utf-8") -> None:
        async with AsyncFiles._semaphore:
            try:
                async with aiofiles.open(path, "w", encoding=encoding) as file:
                    await file.write(data)
            except Exception as e:
                print(f"Error Writing File: {e}")

    @staticmethod
    async def read_json(path: str or Path, default: dict or list = None) -> dict or list:
        async with AsyncFiles._semaphore:
            try:
                async with aiofiles.open(path, "r") as file:
                    return json.loads(await file.read())
            except (FileNotFoundError, json.JSONDecodeError):
                default = {} if default is None else default

        return default

    @staticmethod
    async def write_json(path: str or Path, data: dict or list, indent: int = 4, encoding: str = "utf-8") -> None:
        if isinstance(path, str):
            path = Path(path)

        if not path.suffix == ".json":
            path = path.with_suffix(".json")

        async with AsyncFiles._semaphore:
            try:
                async with aiofiles.open(path, "w", encoding=encoding) as file:
                    await file.write(json.dumps(data, indent=indent))
            except Exception as e:
                print(f"Error Writing JSON: {e}")
