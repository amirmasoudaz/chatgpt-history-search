# main.py

import asyncio
from datetime import datetime
import os
from pathlib import Path

from blake3 import blake3
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from scipy.spatial.distance import cosine as cosine_similarity
from tabulate import tabulate

from gpt import OpenAIClient
from files import AsyncFiles


class ChatGPTSearchEngine:
    def __init__(self):
        self.root = Path().cwd()

        self._paths = self.get_paths()
        self._configs = self.get_configs()

        self._completions_gpt = OpenAIClient(
            model_name=self._configs["chat_model"],
            cache_dir=self._paths["dirs"]["completions_cache"]
        )
        self._embeddings_gpt = OpenAIClient(
            model_name=self._configs["embedding_model"],
            cache_dir=self._paths["dirs"]["embeddings_cache"]
        )

        self.msg_to_ignore = []
        self.indexed_data = {}
        self.vector_cache = {}
        self.search_cache = {}
        self.vector_data = None

    @staticmethod
    def _get_env_variable(key, default=None, required=False, var_type=None):
        var_type = var_type or str
        value = os.environ.get(key, default)
        if required and value is None:
            raise EnvironmentError(f"Missing required environment variable: {key}")
        if value is not None and var_type != str:
            try:
                value = var_type(value)
            except ValueError:
                raise ValueError(f"Environment variable {key} must be of type {var_type.__name__}")
        return value

    def get_paths(self):
        if self._paths is None:
            load_dotenv(find_dotenv("config_paths.env", usecwd=True))
            dirs = {
                "exported": os.path.join(self.root, "data", self._get_env_variable("DIR_EXPORTED")),
                "processed": os.path.join(self.root, "data", self._get_env_variable("DIR_PROCESSED")),
                "embeddings_cache": os.path.join(self.root, "data", self._get_env_variable("DIR_EMBEDDINGS_CACHE")),
                "completions_cache": os.path.join(self.root, "data", self._get_env_variable("DIR_COMPLETIONS_CACHE")),
                "search_cache": os.path.join(self.root, "data", self._get_env_variable("DIR_SEARCH_CACHE"))
            }
            files = {
                "exported": os.path.join(dirs["exported"], self._get_env_variable("FILE_EXPORTED")),
                "index": os.path.join(dirs["processed"], self._get_env_variable("FILE_INDEX")),
                "msg_cache": os.path.join(dirs["processed"], self._get_env_variable("FILE_MSG_CACHE")),
                "vector_cache": os.path.join(dirs["processed"], self._get_env_variable("FILE_VECTOR_CACHE")),
                "vector_data": os.path.join(dirs["processed"], self._get_env_variable("FILE_VECTOR_DATA")),
                "msg_to_ignore": os.path.join(self.root, "data", self._get_env_variable("FILE_MSG_TO_IGNORE"))
            }
            for key, path in dirs.items():
                if key == "exported":
                    continue
                os.makedirs(path, exist_ok=True)
            self._paths = {
                "dirs": dirs,
                "files": files
            }
        return self._paths.copy()

    def get_configs(self):
        if self._configs is None:
            load_dotenv(find_dotenv("config_app.env", usecwd=True))
            self._configs = {
                "chat_model": self._get_env_variable("CHAT_MODEL"),
                "embedding_model": self._get_env_variable("EMBEDDING_MODEL"),
                "ignore_threshold": self._get_env_variable("IGNORE_THRESHOLD", var_type=int),
                "chunk_break_line": self._get_env_variable("CHUNK_BREAK_LINE", var_type=int),
                "chunk_trim_overlap": self._get_env_variable("CHUNK_TRIM_OVERLAP", var_type=int),
                "search_limit": self._get_env_variable("SEARCH_LIMIT", var_type=int)
            }
        return self._configs.copy()

    @staticmethod
    def justified_print(text, length_thr=120):
        lines = text.split('\n')

        for line in lines:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 > length_thr:
                    print(current_line)
                    current_line = word + " "
                else:
                    current_line += word + " "

            if current_line:
                print(current_line.rstrip())
            else:
                print()

    @staticmethod
    def generate_hash(text):
        return blake3(text.encode('utf-8')).hexdigest()

    async def prepare_conversations(self, updates, exported):
        def get_content():
            content = message["content"]
            content_type = content["content_type"]

            if content_type in ['text', 'multimodal_text']:
                text = ' '.join([m.strip() for m in content["parts"] if isinstance(m, str) and m.strip()])
            elif content_type == "code":
                if content['language'] == 'unknown':
                    content['language'] = 'python' if message['recipient'] == 'python' else 'code'
                text = f"Code Snippet: {content['language']}\n\n{content['text']}"
            elif content_type == 'execution_output':
                text = f"Execution Output: {content['text']}"
            elif content_type in ['tether_browsing_display', 'tether_quote']:
                if message['metadata'].get('command') == 'context_stuff':
                    text = f"Context Stuff\n\nTitle: {content['domain']}\n\n{content['text']}"
                elif message["metadata"].get('_cite_metadata'):
                    query = ' '.join([s for s in message['metadata']['args'] if isinstance(s, str)])
                    if message['author']['name'] == 'browser':
                        text = f"Web Browsing Results"
                        text += f"\n\nSearch Query: {query}" if query else ""
                        for r in message['metadata']['_cite_metadata']['metadata_list']:
                            text += f"\n\nType: {r['type']}\nURL: {r['url']}\nTitle: {r['title']}\nResult: {r['text']}"
                    elif message['author']['name'] == 'myfiles_browser':
                        text = f"Files Browsing Results"
                        text += f"\n\nSearch Query: {query}" if query else ""
                        for r in message['metadata']['_cite_metadata']['metadata_list']:
                            text += f"\n\nType: {r['type']}\nName: {r['name']}\nResult: {r['text']}"
                    else:
                        return ''
                else:
                    return ''
            elif content_type in ['system_error']:
                return ''
            else:
                print(f"Unknown message content type: {content_type}")
                return ''

            if not text or len(text) < self._configs["ignore_threshold"]:
                return ''

            return text

        def get_chunks():
            break_limit, overlap = self._configs["chunk_break_line"], self._configs["chunk_trim_overlap"]
            try:
                tokenized = self._embeddings_gpt.tokenizer.tokenize(message_content)
                n_tokens = len(tokenized)
                n_segments = max(1, round(n_tokens / break_limit))

                if abs(n_tokens - break_limit) <= abs(n_tokens / n_segments - break_limit):
                    return [message_content]

                optimal = n_tokens // n_segments
                segments = []
                for i in range(0, n_tokens, optimal):
                    start = i - overlap if i > overlap else 0
                    end = i + optimal + overlap if (i + optimal + overlap <= n_tokens) else n_tokens
                    segments.append(tokenized[start:end])

                if len(segments) > 1 and len(segments[-1]) < optimal:
                    segments[-2].extend(segments[-1])
                    segments.pop()

                return [self._embeddings_gpt.tokenizer.stringify(segment) for segment in segments]
            except Exception as e:
                print(f"Error processing text: {e}")
                return [message_content]

        msg_cache = await AsyncFiles.read_files(self._paths["files"]["msg_cache"], default={})
        for conversation in exported[::-1]:
            conversation_id = conversation.get("conversation_id")
            if not conversation_id:
                continue

            if conversation_id not in updates:
                continue

            conversation_title = ' '.join(conversation.get("title", "").split())
            if not conversation_title:
                conversation_title = f"Untitled Chat"

            messages = []
            for message in conversation["mapping"].values():
                message = message.get("message")
                if not message:
                    continue

                role = message["author"]["role"]
                if role == "system":
                    continue

                if message["status"] != "finished_successfully":
                    continue

                message_content = get_content()
                if not message_content:
                    continue

                message_segments = get_chunks()

                for msg in message_segments:
                    msg_hash = self.generate_hash(msg)
                    if not msg_cache.get(msg_hash):
                        msg_cache[msg_hash] = {
                            "content": msg, "addresses": [[conversation_id, len(messages)]]
                        }
                    elif [conversation_id, len(messages)] not in msg_cache[msg_hash]["addresses"]:
                        msg_cache[msg_hash]["addresses"].append([conversation_id, len(messages)])

                model = message["metadata"].get("model_slug", "gpt") if role == "assistant" else "user"
                messaged_at = datetime.fromtimestamp(message["create_time"]).strftime("%Y-%m-%d %H:%M:%S")
                messages.append({
                    "context": {
                        "role": role,
                        "content": message_content,
                    },
                    "metadata": {
                        "model": model,
                        "created_at": messaged_at,
                        "message_index": len(messages),
                        "conversation_id": conversation_id,
                        "conversation_title": conversation_title,
                    }
                })

            if not messages:
                self.msg_to_ignore.append(conversation_id)
                continue

            total_raw_messages = len(conversation["mapping"].values())
            created_at = datetime.fromtimestamp(conversation.get("create_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
            conversation_url = "https://chatgpt.com/c/" + conversation_id
            self.indexed_data[conversation_id] = {
                "messages": messages,
                "created_at": created_at,
                "conversation_id": conversation_id,
                "conversation_title": conversation_title,
                "total_processed_messages": len(messages),
                "total_raw_messages": total_raw_messages,
                "conversation_url": conversation_url
            }

        print(f"Total Chats: {len(self.indexed_data)} - Total Msg Chunks: {len(msg_cache)} -")
        return msg_cache

    async def generate_embeddings(self, msg_cache):
        tokens = []
        vector_cache = await AsyncFiles.read_json(self._paths["files"]["vector_cache"], default={})
        tasks = []
        for msg_hash, msg in msg_cache.items():
            if msg.get("embedding"):
                continue
            else:
                if vector_cache.get(msg_hash) and vector_cache[msg_hash].get("output"):
                    msg["embedding"] = vector_cache[msg_hash]["output"]
                else:
                    tasks.append(self._embeddings_gpt.get_response(
                        input=msg["content"],
                        identifier=msg_hash,
                        cache_response=True
                    ))
                    tokens.append(self._embeddings_gpt.tokenizer.count_tokens(msg["content"]))

        if not tokens:
            print(f"- No New API Calls Required - Data Already Cached -")
        else:
            print(f"- New API Calls: {len(tokens)} - Tokens: {sum(tokens)} -", end=" ")
            print(f"Cost: ${round(sum(tokens) / 1000 * self._embeddings_gpt.spec.usage_costs['input'], 4)} -", end=" ")
            print(f"Model: {self._embeddings_gpt.model_name} -", end=" ")
            results = await self._embeddings_gpt.run_batch(tasks)
            print("Fetched Successfully -")

            if results:
                for result in results:
                    if result["output"]:
                        msg_cache[result["identifier"]]["embedding"] = result["output"]
                        os.remove(os.path.join(self._paths["dirs"]["vector_cache"], f"{result['identifier']}.json"))
                        vector_cache[result["identifier"]] = result
                    else:
                        print(f"- Failed to Embed: {result['identifier']}")

        return pd.DataFrame([
            {"hash": msg_hash, **msg}
            for msg_hash, msg in msg_cache.items()
        ]), vector_cache

    async def search(self, query, identifier, limit):
        if identifier in self.search_cache:
            return self.search_cache[identifier]

        result = await self._embeddings_gpt.get_response(input=query, identifier=identifier, cache_response=True)
        data = [
            (row["addresses"], row["hash"], 1 - cosine_similarity(result["output"], row["embedding"]))
            for i, row in self.vector_data.iterrows()
        ]

        self.search_cache[identifier] = result
        self.search_cache[identifier]["search_query"] = query
        search_results = sorted(data, key=lambda x: x[2], reverse=True)
        result_addresses = []
        for i, (addresses, msg_hash, score) in enumerate(search_results):
            for address in addresses:
                if address[0] not in result_addresses:
                    result_addresses.append(address[0])
            if len(result_addresses) >= limit:
                break

        self.search_cache[identifier]["results"] = result_addresses[:limit]
        file_path = os.path.join(self._paths["dirs"]["search_cache"], f"{identifier}.json")
        await AsyncFiles.write_json(file_path, self.search_cache[identifier])

        return self.search_cache[identifier]

    async def prep_logic(self):
        self.msg_to_ignore = await AsyncFiles.read_json(self._paths["files"]["msg_to_ignore"], default=self.msg_to_ignore)
        self.indexed_data = await AsyncFiles.read_json(self._paths["files"]["index"], default=self.indexed_data)
        self.search_cache = await AsyncFiles.read_files(self._paths["dirs"]["search_cache"], dtype="json", default=self.search_cache)
        self.vector_data = pd.read_pickle(self._paths["files"]["vector_data"])


        exported = await AsyncFiles.read_json(self._paths["files"]["exported"], default={})
        if not self.indexed_data and not exported:
            raise FileNotFoundError(f"- Exported JSON File Not Found - Path: {self._paths['files']['exported']}")

        updates = []
        for conversation in exported:
            conversation_id = conversation.get("conversation_id")
            if not conversation_id or conversation_id in self.msg_to_ignore:
                continue

            if not self.indexed_data.get(conversation_id):
                updates.append(conversation_id)
            else:
                total_raw_messages = len(conversation["mapping"].values())
                if self.indexed_data[conversation_id]["total_raw_messages"] != total_raw_messages:
                    updates.append(conversation_id)

        if updates:
            print(f"- Processing Exported Chats - New Chats: {len(updates)} -", end=" ")
            msg_cache = await self.prepare_conversations(updates, exported)
            self.vector_data, vector_cache = await self.generate_embeddings(msg_cache=msg_cache)

            print(f"- Finalizing and Storing Processed Data -", end=" ")
            await AsyncFiles.write_json(self._paths["files"]["index"], self.indexed_data)
            await AsyncFiles.write_json(self._paths["files"]["msg_cache"], msg_cache)
            await AsyncFiles.write_json(self._paths["files"]["vector_cache"], vector_cache)
            await AsyncFiles.write_json(self._paths["files"]["msg_to_ignore"], self.msg_to_ignore)
            self.vector_data.to_pickle(self._paths["files"]["vector_data"], protocol=4)
            print(f"Done -")

    async def chat_logic(self, results, result_index, identifier):
        conversation_title = results["results"][result_index - 1]
        context = self.indexed_data[conversation_title].copy()
        context_str = ""
        context_list = []
        for message in context["messages"]:
            if message["context"]["role"] != "user":
                message["context"]["role"] = "assistant"
            context_str += f"- {message['context']['role'].title()}: {message['context']['content']}\n\n-----\n\n"
            context_list.append(message['context'])

        token_count = self._completions_gpt.tokenizer.count_tokens(context_list)
        cost = round(token_count / 1000 * self._completions_gpt.spec.usage_costs['input'], 4)
        print(f"\n\n- {context['conversation_title']} -")
        print(f"- Length: {len(context['messages'])} Messages - Length: {token_count} Tokens -")
        print(f"- API Input Cost: ~${cost}+ Per Prompt Using {self._completions_gpt.model_name} Model -")
        print(f"- ChatGPT URL: {context['conversation_url']} -\n\n")
        self.justified_print(context_str[:-1])

        while True:
            user_query = input("- User (0 to Return): ")
            if user_query == "0":
                break
            context_list.append({"role": "user", "content": user_query})

            response = await self._completions_gpt.get_response(messages=context_list, identifier=identifier)

            context_list.append({"role": "assistant", "content": response["output"]})
            self.justified_print(f"\n-----\n\n- Assistant: {response['output']}")
            self.indexed_data[conversation_title]["messages"].extend([context_list[-2], context_list[-1]])

        # Add the new messages into the index and generate embeddings
        await AsyncFiles.write_json(self._paths["files"]["index"], self.indexed_data)

    async def search_logic(self):
        self._completions_gpt.cache_dir = self._paths["dirs"]["search_cache"]
        self._embeddings_gpt.cache_dir = self._paths["dirs"]["search_cache"]

        while True:
            query = input("- Search Query (0 to Exit): ")
            if query == "0":
                break
            try:
                page_size = int(input("- Page Size (Default: 10): ") or self._configs["search_limit"])
            except ValueError:
                page_size = self._configs["search_limit"]

            identifier = self.generate_hash(query)
            results = await self.search(query, identifier, limit=page_size)

            print(f"- Search Results for '{query}':")
            table = []
            for i, address in enumerate(results["results"], start=1):
                info = self.indexed_data[address]
                table.append([i, info["conversation_title"], info["created_at"], info["conversation_url"]])
                if i >= page_size:
                    break
            print(tabulate(table, headers=["INDEX", "TITLE", "CREATED AT", "URL"], tablefmt="grid"))

            result_index = int(input("\n- Index to Continue Chat With (0 to Return): "))
            if result_index == 0:
                continue
            else:
                await self.chat_logic(results, result_index, identifier)

    async def main(self):
        await self.prep_logic()
        await self.search_logic()


if __name__ == "__main__":
    processor = ChatGPTSearchEngine()
    asyncio.run(processor.main())
