import asyncio
import json
import os

import aiofiles
import pandas as pd


class FileTools:
    def __init__(self, max_concurrent_files=100):
        self.semaphore = asyncio.Semaphore(max_concurrent_files)

    def read_dir_contents(self, path: str, dtype: str = "json", default=None) -> dict:
        """
        Read all files in a directory with a specific dtype

        :param path: Path to the directory
        :param dtype: Data type of the files to read  (json, txt, csv, etc.)
        :param default: Default value to return if file is not found

        :return: Dictionary of file contents with file name (without the dtype) as key and content as value
        """

        contents = {}
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(dtype):
                    ident = file.replace(f".{dtype}", "")

                    if dtype == "json":
                        contents[ident] = self.read_json(os.path.join(path, file), default=default)
                    elif dtype in ["csv", "xlsx", "pkl"]:
                        contents[ident] = self.read_df(os.path.join(path, file), dtype, default=default)
                    else:
                        contents[ident] = self.read_file(os.path.join(path, file), default=default)

        return contents

    async def read_dir_contents_async(self, path: str, dtype: str = "json", default=None) -> dict:
        """
        Read all files in a directory with a specific dtype asynchronously

        :param path: Path to the directory
        :param dtype: Data type of the files to read  (json, txt, etc.) [NOT SUPPORTED FOR DATAFRAMES]
        :param default: Default value to return if file is not found

        :return: Dictionary of file contents with file name (without the dtype) as key and content as value
        """

        tasks, idents = [], []
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(dtype):
                    idents.append(file.replace(f".{dtype}", "") if dtype != "all" else file)

                    if dtype == "json":
                        tasks.append(self.read_json_async(os.path.join(path, file), default=default))
                    else:
                        tasks.append(self.read_file_async(os.path.join(path, file), default=default))

        results = await asyncio.gather(*tasks)
        contents = {ident: content for ident, content in zip(idents, results)}

        return contents

    def write_contents_to_dir(self, path: str, contents: dict, dtype: str) -> None:
        """
        Write contents to a directory with a specific dtype

        :param path: Path to the directory
        :param contents: Dictionary of file contents with file name as key and content as value
        :param dtype: Data type of the files to write  (json, txt, csv, etc.)

        :return: None
        """

        if not os.path.exists(path):
            os.makedirs(path)

        for ident, content in contents.items():
            file_path = os.path.join(path, f"{ident}.{dtype}")
            if dtype == "json":
                self.write_json(file_path, content)
            elif dtype in ["csv", "xlsx", "pkl"]:
                self.write_df(file_path, content, dtype)
            else:
                self.write_file(file_path, content)

    async def write_contents_to_dir_async(self, path: str, contents: dict, dtype: str) -> None:
        """
        Write contents to a directory with a specific dtype asynchronously

        :param path: Path to the directory
        :param contents: Dictionary of file contents with file name as key and content as value
        :param dtype: Data type of the files to write  (json, txt, etc.) [NOT SUPPORTED FOR DATAFRAMES]

        :return: None
        """

        if not os.path.exists(path):
            os.makedirs(path)

        tasks = []
        for ident, content in contents.items():
            file_path = os.path.join(path, f"{ident}.{dtype}")
            if dtype == "json":
                tasks.append(self.write_json_async(file_path, content))
            else:
                tasks.append(self.write_file_async(file_path, content))

        await asyncio.gather(*tasks)

    def read_file(self, path: str, default=None) -> str:
        """
        Read a file

        :param path: Path to the file
        :param default: Default value to return if file is not found

        :return: content
        """

        try:
            with open(path, "r") as file:
                return file.read()
        except FileNotFoundError:
            self.write_file(path, data="" if default is None else default)

        return default

    @staticmethod
    def write_file(path: str, data: str) -> None:
        """
        Write data to a file

        :param path: Path to the file
        :param data: Data to write

        :return: None
        """

        try:
            with open(path, "w") as file:
                file.write(data)
        except Exception as e:
            print(f"Error Writing File: {e}")

    async def read_file_async(self, path: str, default=None) -> str:
        """
        Read a file asynchronously

        :param path: Path to the file
        :param default: Default value to return if file is not found

        :return: content
        """

        async with self.semaphore:
            try:
                async with aiofiles.open(path, "r") as file:
                    return await file.read()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error Reading File: {e}")

            return default

    async def write_file_async(self, path: str, data: str) -> None:
        """
        Write data to a file asynchronously

        :param path: Path to the file
        :param data: Data to write

        :return: None
        """

        async with self.semaphore:
            try:
                async with aiofiles.open(path, "w") as file:
                    await file.write(data)
            except Exception as e:
                print(f"Error Writing File: {e}")

    def read_json(self, path: str, default: dict or list = None) -> dict or list:
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
            default = {} if default is None else default
            self.write_json(path, default, indent=4)

        return default

    @staticmethod
    def write_json(path: str, data: list or dict, indent: int = 4) -> None:
        """
        Write data to a JSON file

        :param path: Path to the JSON file
        :param data: Data to write
        :param indent: Indentation level

        :return: None
        """
        if not path.endswith(".json"):
            path += ".json"

        try:
            with open(path, "w") as file:
                json.dump(data, file, indent=indent)
        except Exception as e:
            print(f"Error Writing JSON: {e}")

    async def read_json_async(self, path: str, default: dict or list = None) -> dict or list:
        """
        Read a JSON file asynchronously

        :param path: Path to the JSON file
        :param default: Default value to return if file is not found or corrupted

        :return: JSON data
        """

        async with self.semaphore:
            try:
                async with aiofiles.open(path, "r") as file:
                    return json.loads(await file.read())
            except (FileNotFoundError, json.JSONDecodeError):
                default = {} if default is None else default
                await self.write_json_async(path, default, indent=4)

            return default

    async def write_json_async(self, path: str, data: dict, indent: int = 4) -> None:
        """
        Write data to a JSON file asynchronously

        :param path: Path to the JSON file
        :param data: Data to write
        :param indent: Indentation level

        :return: None
        """
        if not path.endswith(".json"):
            path += ".json"

        async with self.semaphore:
            try:
                async with aiofiles.open(path, "w") as file:
                    await file.write(json.dumps(data, indent=indent))
            except Exception as e:
                print(f"Error Writing JSON: {e}")

    @staticmethod
    def read_df(path: str, dtype: str = "csv", default=None) -> pd.DataFrame:
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
    def write_df(path: str, data: pd.DataFrame, dtype: str = "csv") -> None:
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
