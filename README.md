# ChatGPT History Search Engine

This Python project creates a search engine for OpenAI's ChatGPT conversation history, enabling users to search, view, and interact with archived conversations. It leverages text embeddings for semantic search capabilities.

## Features

- Load and update conversation and index data dynamically from files.
- Generate and store text embeddings for efficient semantic search.
- Provide interactive search and conversation replay in the terminal.

## Preparation

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Export chat history and add them to the project:**
   - Navigate to the ChatGPT website and go to Settings -> Data controls -> Export data.
   - Wait for the data to be prepared and download the zip file.
   - Extract the `conversations.json` from the zip file and place it in the `data/exported` directory within the project.

3. **Add OpenAI API key for API access:**
   Add your OpenAI API key to the `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the program:**
   - Execute the program with the following command:
     ```bash
     python main.py
     ```
   - On the first run, the program will reformat `conversations.json`, prepare messages for text embeddings, and prompt you to generate all embeddings. After indexing, these embeddings are stored for future use.
   - To update the chat history with a new `conversations.json`, replace the old one and set the `update_db` parameter to True when running the program. Setting it to False after updates can help avoid unnecessary background processing and potential performance degradation.

## Using the Search Results

After executing a search, you'll receive a list of chat results with their titles, URLs, and creation dates. You can either:

- Continue the conversation using the ChatGPT API in the Python console. Follow the interactive prompts to input your query and receive responses.
- Use the provided URLs to view and continue the conversation directly on the ChatGPT website.

Here's a sample output of the console:

```
- Search Query (0 to Exit): FastAPI
- Search Results for FastAPI:
+---------+----------------------------------------------------+------------------------------------------------------------+---------------------+
|   INDEX | TITLE                                              | URL                                                        | CREATED AT          |
+=========+====================================================+============================================================+=====================+
|       1 | Chat 277 - Async Chatbot Solution.                 | https://chatgpt.com/c/800693d5-9715-4bf9-9d08-251126da782c | 2023-08-02 00:25:25 |
+---------+----------------------------------------------------+------------------------------------------------------------+---------------------+
|       2 | Chat 278 - FastAPI, Flask, and concurrency.        | https://chatgpt.com/c/9354259f-99e3-4aa7-bef7-b53fe30965a1 | 2023-08-02 01:02:01 |
+---------+----------------------------------------------------+------------------------------------------------------------+---------------------+
|       3 | Chat 1422 - Uvicorn: ASGI Server Implementation    | https://chatgpt.com/c/f5394fec-28fc-4f57-aea6-994b3d165298 | 2024-04-28 23:31:55 |
+---------+----------------------------------------------------+------------------------------------------------------------+---------------------+
- Index to Continue With (0 to Exit): 2

Chat Title: Chat 278 - FastAPI, Flask, and concurrency.

- User: ...
- Assistant: ...
.
.
.

- User (0 to Exit): Can you explain further about the differences between FastAPI and Flask?
- Context Has 9236 Tokens - API Input Cost: ~$0.0046 - Proceed? (y/n): y

Assistant: ...
```