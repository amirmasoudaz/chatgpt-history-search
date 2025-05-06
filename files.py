# src/utils/files.py

import asyncio
import json
import shutil
from pathlib import Path

import aiofiles
import pandas as pd


class Paths:
    @staticmethod
    def get_cpd(filename="__init__.py"):
        current_path = Path(__name__).resolve().parent
        if (current_path / filename).exists():
            return current_path

        for parent in current_path.parents:
            if (parent / filename).exists():
                return parent
        return None

    @staticmethod
    def rename_file(src: str or Path, dest: str or Path) -> None:
        """
        Rename a file

        :param src: Source path of the file
        :param dest: Destination path of the file

        :return: None
        """
        try:
            shutil.move(src, dest)
        except Exception as e:
            print(f"Error Renaming File: {e}")

    @staticmethod
    def copy_file(src: str or Path, dest: str or Path) -> None:
        """
        Copy a file from one location to another

        :param src: Source path of the file
        :param dest: Destination path of the file

        :return: None
        """
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            print(f"Error Copying File: {e}")

    @staticmethod
    def move_file(src: str, dest: str or Path) -> None:
        """
        Move a file from one location to another

        :param src: Source path of the file
        :param dest: Destination path of the file

        :return: None
        """
        try:
            shutil.move(src, dest)
        except Exception as e:
            print(f"Error Moving File: {e}")

    @staticmethod
    def copy_dir(src: str or Path, dest: str or Path) -> None:
        """
        Copy a directory from one location to another

        :param src: Source path of the directory
        :param dest: Destination path of the directory

        :return: None
        """
        try:
            shutil.copytree(src, dest)
        except Exception as e:
            print(f"Error Copying Directory: {e}")

    @staticmethod
    def normalize_path(path: str or Path) -> str or Path:
        forbidden_chars = ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]
        for char in forbidden_chars:
            if isinstance(path, Path):
                path = path.with_name(path.name.replace(char, "-"))
            elif isinstance(path, str):
                path = path.replace(char, "-")
        return path


class SyncFiles(Paths):
    @staticmethod
    def read_files(path: str or Path, dtype: str = "json", default=None) -> dict:
        """
        Read all files in a directory with a specific dtype

        :param path: Path to the directory
        :param dtype: Data type of the files to read  (json, txt, csv, etc.)
        :param default: Default value to return if file is not found

        :return: Dictionary of file contents with file name (without the dtype) as key and content as value
        """
        contents = {}
        if isinstance(path, str):
            path = Path(path)
        if path.exists() and path.is_dir():
            for file in path.iterdir():
                if file.suffix == f".{dtype}":
                    ident = file.stem

                    if dtype == "json":
                        contents[ident] = SyncFiles.read_json(file, default=default)
                    elif dtype in ["csv", "xlsx", "pkl"]:
                        contents[ident] = SyncFiles.read_df(file, dtype, default=default)
                    else:
                        contents[ident] = SyncFiles.read_file(file, default=default)

        return contents

    @staticmethod
    def write_files(path: str or Path, contents: dict, dtype: str, encoding: str = "utf-8") -> None:
        """
        Write contents to a directory with a specific dtype

        :param path: Path to the directory
        :param contents: Dictionary of file contents with file name as key and content as value
        :param dtype: Data type of the files to write  (json, txt, csv, etc.)
        :param encoding: Encoding to use when writing the file (default: utf-8)

        :return: None
        """
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        for ident, content in contents.items():
            file_path = path / f"{ident}.{dtype}"
            if dtype in ["csv", "xlsx", "pkl"]:
                SyncFiles.write_df(file_path, content, dtype)
            elif dtype == "json":
                SyncFiles.write_json(file_path, content, encoding=encoding)
            else:
                SyncFiles.write_file(file_path, content, encoding=encoding)

    @staticmethod
    def read_file(path: str or Path, default=None, route=False, encoding=None) -> any:
        """
        Read a file

        :param path: Path to the file
        :param default: Default value to return if file is not found
        :param route: Route to the correct function
        :param encoding: Encoding to use when reading the file

        :return: content
        """
        if route:
            if isinstance(path, str):
                path = Path(path)

            dtype = path.suffix[1:]
            if dtype == "json":
                return SyncFiles.read_json(path, default=default)
            elif dtype in ["csv", "xlsx", "pkl"]:
                return SyncFiles.read_df(path, dtype, default=default)
            else:
                return SyncFiles.read_file(path, default=default, route=False, encoding=encoding)
        try:
            with open(path, "r", encoding=encoding) as file:
                return file.read()
        except FileNotFoundError:
            return default

    @staticmethod
    def write_file(path: str or Path, data: any, mode: str = "w", encoding: str = "utf-8") -> None:
        """
        Write data to a file

        :param path: Path to the file
        :param data: Data to write
        :param mode: Method to use when writing the file (default: w)
        :param encoding: Encoding to use when writing the file (default: utf-8)

        :return: None
        """
        if "b" in mode:
            encoding = None

        try:
            with open(path, mode, encoding=encoding) as file:
                file.write(data)
        except Exception as e:
            print(f"Error Writing File: {e}")

    @staticmethod
    def read_json(path: str or Path, default: dict or list = None) -> dict or list:
        """
        Read a JSON file

        :param path: Path to the JSON file
        :param default: Default value to return if file is not found or corrupted

        :return: JSON data
        """

        try:
            with open(path, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def write_json(path: str or Path, data: list or dict, indent: int = 4, encoding: str = "utf-8") -> None:
        """
        Write data to a JSON file

        :param path: Path to the JSON file
        :param data: Data to write
        :param indent: Indentation level
        :param encoding: Encoding to use when writing the file (default: utf-8)

        :return: None
        """
        if isinstance(path, str):
            path = Path(path)

        if not path.suffix == ".json":
            path = path.with_suffix(".json")

        try:
            with open(path, "w", encoding=encoding) as file:
                file.write(json.dumps(data, indent=indent))
        except Exception as e:
            print(f"Error Writing JSON: {e}")

    @staticmethod
    def read_df(path: str or Path, dtype: str = "csv", default=None) -> pd.DataFrame:
        """
        Read a dataframe from a file

        :param path: Path to the file
        :param dtype: Data type of the file  (csv, pkl, xlsx)
        :param default: Default value to return if file is not found

        :return: Pandas DataFrame
        """
        try:
            if dtype == "csv":
                return pd.read_csv(path)
            elif dtype == "pkl":
                return pd.read_pickle(path)
            elif dtype == "xlsx":
                return pd.read_excel(path)
            else:
                raise ValueError(f"Invalid Dataframe dtype: {dtype}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error: {e}")

        return default

    @staticmethod
    def write_df(path: str or Path, data: pd.DataFrame, dtype: str = "csv") -> None:
        """
        Write a dataframe to a file

        :param path: Path to the file
        :param data: Pandas DataFrame
        :param dtype: Data type of the file  (csv, pkl, xlsx)

        :return: None
        """
        try:
            if dtype == "csv":
                data.to_csv(path, index=False)
            elif dtype == "pkl":
                data.to_pickle(path)
            elif dtype == "xlsx":
                data.to_excel(path, index=False)
            else:
                raise ValueError(f"Invalid Dataframe dtype: {dtype}")
        except Exception as e:
            print(f"Error Writing CSV: {e}")


class AsyncFiles(Paths):
    concurrent_operations = 300
    _semaphore = asyncio.Semaphore(concurrent_operations)

    @staticmethod
    async def read_files(path: str or Path, dtype: str = "json", default=None) -> dict:
        """
        Read all files in a directory with a specific dtype asynchronously

        :param path: Path to the directory
        :param dtype: Data type of the files to read  (json, txt, etc.) [NOT SUPPORTED FOR DATAFRAMES]
        :param default: Default value to return if file is not found

        :return: Dictionary of file contents with file name (without the dtype) as key and content as value
        """
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
        """
        Write contents to a directory with a specific dtype asynchronously

        :param path: Path to the directory
        :param contents: Dictionary of file contents with file name as key and content as value
        :param dtype: Data type of the files to write  (json, txt, etc.) [NOT SUPPORTED FOR DATAFRAMES]
        :param encoding: Encoding to use when writing the file (default: utf-8)

        :return: None
        """
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
        """
        Read a file asynchronously

        :param path: Path to the file
        :param default: Default value to return if file is not found
        :param route: Route to the correct function
        :param encoding: Encoding to use when reading the file

        :return: content
        """
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
        """
        Write data to a file asynchronously

        :param path: Path to the file
        :param data: Data to write
        :param encoding: Encoding to use when writing the file (default: utf-8)

        :return: None
        """
        async with AsyncFiles._semaphore:
            try:
                async with aiofiles.open(path, "w", encoding=encoding) as file:
                    await file.write(data)
            except Exception as e:
                print(f"Error Writing File: {e}")

    @staticmethod
    async def read_json(path: str or Path, default: dict or list = None) -> dict or list:
        """
        Read a JSON file asynchronously

        :param path: Path to the JSON file
        :param default: Default value to return if file is not found or corrupted

        :return: JSON data
        """
        async with AsyncFiles._semaphore:
            try:
                async with aiofiles.open(path, "r") as file:
                    return json.loads(await file.read())
            except (FileNotFoundError, json.JSONDecodeError):
                default = {} if default is None else default

        return default

    @staticmethod
    async def write_json(path: str or Path, data: dict or list, indent: int = 4, encoding: str = "utf-8") -> None:
        """
        Write data to a JSON file asynchronously

        :param path: Path to the JSON file
        :param data: Data to write
        :param indent: Indentation level
        :param encoding: Encoding to use when writing the file (default: utf-8)

        :return: None
        """
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
