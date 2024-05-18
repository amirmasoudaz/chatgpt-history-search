import asyncio
from datetime import datetime
import hashlib
import os

import pandas as pd
from scipy.spatial.distance import cosine as cosine_similarity
from tabulate import tabulate

from core import Core
from utilities.helpers import justified_print
from utilities.files import IOFiles


class ChatGPTSearchEngine(Core):
    def __init__(self):
        super().__init__()

    @staticmethod
    def generate_hash(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def load_data(self, update_db=False):
        if not os.path.exists(self.paths["file"]["index"]):
            update_db = True  # Force Update

        self.index = IOFiles.read_json(self.paths["file"]["index"])
        self.vector = IOFiles.read_df(self.paths["file"]["vector"], dtype="pickle")
        self.search_cache = IOFiles.read_dir_contents(self.paths["dir"]["logs"], dtype="json")

        if update_db:
            self.exported = IOFiles.read_json(self.paths["file"]["exported"])
            if not self.exported:
                raise FileNotFoundError(f"- Exported JSON File Not Found - Path: {self.paths["file"]["exported"]}")

            self.cache = IOFiles.read_json(self.paths["file"]["cache"])
            self.vector_cache = IOFiles.read_dir_contents(self.paths["dir"]["cache"], dtype="json")

    def save_data(self):
        print(f"- Saving Processed Data -", end=" ")

        IOFiles.write_json(self.paths["file"]["index"], self.index)
        IOFiles.write_json(self.paths["file"]["cache"], self.cache)
        IOFiles.write_df(self.paths["file"]["vector"], self.vector, dtype="pickle")

        print(f"Done -")

    def prepare_conversations(self):
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
                text = f"Execution Output: {content["text"]}"
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

            if not text or len(text) < self.IGNORE_THRESHOLD:
                return ''

            return text

        def get_chunks():
            breaklimit, overlap = self.CHUNK_BREAK_LINE, self.CHUNK_TRIM_OVERLAP
            try:
                tokenized = self.gpt_client.calculator.tokenize(message_content)
                n_tokens = len(tokenized)
                n_segments = max(1, round(n_tokens / breaklimit))

                if abs(n_tokens - breaklimit) <= abs(n_tokens / n_segments - breaklimit):
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

                return [self.gpt_client.calculator.stringify(segment) for segment in segments]
            except Exception as e:
                print(f"Error processing text: {e}")
                return [message_content]

        for idx, conversation in enumerate(self.exported[::-1]):
            title = ' '.join(conversation.get("title", "").split())
            title = f"Chat {idx + 1} - {title}" if title else f"Chat {idx + 1}"

            created_at = datetime.fromtimestamp(conversation.get("create_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
            conversation_id = conversation.get("conversation_id", "")

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
                    if msg_hash not in self.cache:
                        self.cache[msg_hash] = {
                            "content": msg, "addresses": [[title, len(messages)]]
                        }
                    elif [title, len(messages)] not in self.cache[msg_hash]["addresses"]:
                        self.cache[msg_hash]["addresses"].append([title, len(messages)])

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
                        "conversation_id": conversation_id,
                        "message_index": len(messages),
                        "conversation_title": title,
                    }
                })

            if not messages:
                continue

            self.index[title] = {
                "messages": messages,
                "created_at": created_at,
                "conversation_id": conversation_id,
                "URL": "https://chatgpt.com/c/" + conversation_id
            }

    async def generate_embeddings(self):
        tokens = []
        for msg_hash, msg in self.cache.items():
            if msg.get("embedding"):
                continue
            else:
                if msg_hash in self.vector_cache:
                    msg["embedding"] = self.vector_cache[msg_hash]["output"]
                else:
                    self.gpt_client.add_request(
                        context=msg["content"],
                        identifier=msg_hash,
                        engine="embedding")
                    tokens.append(self.gpt_client.calculator.count_tokens(msg["content"]))

        print(f"- {len(self.index)} Conversations - {len(self.cache)} Rows -", end=" ")

        if not tokens:
            print(f"Up-to-Date")
            return

        print(f"{len(tokens)} API Calls - {sum(tokens)} Tokens -", end=" ")
        cost = round(sum(tokens) / 1000 * self.gpt_client.embedding_model['cost']['input'], 4)
        print(f"API Cost: ${cost} -")

        if input("- Proceed With Fetching Text Embeddings? (y/n): ").lower() != "y":
            print("- Aborted!")
            return

        print("- Fetching Text Embeddings - ", end=" ")
        results = await self.gpt_client.trigger_requests()
        print("Done -")

        if results:
            for result in results:
                if result["output"]:
                    self.cache[result["identifier"]]["embedding"] = result["output"]
                else:
                    print(f"- Failed to Embed: {result['identifier']}")

            self.vector = pd.DataFrame([
                {"hash": msg_hash, **msg}
                for msg_hash, msg in self.cache.items()
            ])

    async def search(self, query, identifier):
        if identifier in self.search_cache:
            return self.search_cache[identifier]

        result = await self.gpt_client.call_model(
            context=query,
            identifier=identifier,
            engine="embedding")

        data = [
            (row["addresses"], row["hash"], 1 - cosine_similarity(result["output"], row["embedding"]))
            for i, row in self.vector.iterrows()
        ]

        self.search_cache[identifier] = result
        self.search_cache[identifier]["search_query"] = query
        search_results = sorted(data, key=lambda x: x[2], reverse=True)
        result_addresses = []
        for i, (addresses, msg_hash, score) in enumerate(search_results):
            for address in addresses:
                if address[0] not in result_addresses:
                    result_addresses.append(address[0])
            if len(result_addresses) >= self.SEARCH_LIMIT:
                break

        self.search_cache[identifier]["results"] = result_addresses[:self.SEARCH_LIMIT]
        file_path = os.path.join(self.paths["dir"]["logs"], f"{identifier}.json")
        IOFiles.write_json(file_path, self.search_cache[identifier])

        return self.search_cache[identifier]

    async def prep_logic(self):
        print("- Processing Exported Messages -")
        self.prepare_conversations()
        await self.generate_embeddings()
        self.save_data()
        print("- Exported Messages Processed -")

    async def chat_logic(self, results, result_index, identifier):
        conversation_title = results["results"][result_index - 1]
        context = self.index[conversation_title].copy()
        context_str = f"\nChat Title: {conversation_title}\n\n"
        context_list = []
        for message in context["messages"]:
            if message["context"]["role"] != "user":
                message["context"]["role"] = "assistant"
            context_str += f"- {message['context']['role'].title()}: {message['context']['content']}\n\n-----\n\n"
            context_list.append(message['context'])
        justified_print(context_str[:-1])

        while True:
            user_query = input("\n- User (0 to Exit): ")
            if user_query == "0":
                break
            context_list.append({"role": "user", "content": user_query})

            token_count = self.gpt_client.calculator.count_tokens(context_list)
            cost = round(token_count / 1000 * self.gpt_client.chat_model['cost']['input'], 4)
            if input(f"- Context Has {token_count} Tokens - API Input Cost: ~${cost} - Proceed? (y/n): ").lower() != "y":
                print("- Aborted!")
                break

            # Add token streaming
            response = await self.gpt_client.call_model(
                context=context_list,
                identifier=identifier,
                engine="chat")

            context_list.append({"role": "assistant", "content": response["output"]})
            justified_print(f"\n-----\n\n- Assistant: {response["output"]}")
            self.index[conversation_title]["messages"].extend([context_list[-2], context_list[-1]])

        # Add the new messages into the index and generate embeddings
        IOFiles.write_json(self.paths["file"]["index"], self.index)

    async def search_logic(self):
        self.gpt_client.cache_dir = self.paths["dir"]["logs"]

        while True:
            query = input("- Search Query (0 to Exit): ")
            if query == "0":
                break

            identifier = self.generate_hash(query)
            results = await self.search(query, identifier)

            print(f"- Search Results for {query}:")
            table = []
            for i, address in enumerate(results["results"]):
                table.append([i + 1, address, self.index[address]["URL"], self.index[address]["created_at"]])
            print(tabulate(table, headers=["INDEX", "TITLE", "URL", "CREATED AT"], tablefmt="grid"))

            result_index = int(input("- Index to Continue With (0 to Exit): "))
            if result_index == 0:
                continue
            else:
                await self.chat_logic(results, result_index, identifier)

    async def main(self, update_database=False):
        self.load_data(update_database)
        if update_database:
            await self.prep_logic()
        await self.search_logic()


if __name__ == "__main__":
    processor = ChatGPTSearchEngine()
    asyncio.run(processor.main())
