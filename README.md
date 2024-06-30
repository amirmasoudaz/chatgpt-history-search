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
   - On the first run, the program will reformat `conversations.json` and prepare messages for text embeddings.
   ```bash
   - Processing Exported Chats - New Chats: 1632 - Total Chats: 1620 - Total Msg Chunks: 17495 -
   ```
   - Then it will notify you of generating all embeddings.
    ```bash
   - New API Calls: 17495 - Tokens: 9219971 - Cost: $1.1986 - Model: text-embedding-3-large - Fetched Successfully -
    ```
   - After indexing, these embeddings are stored for future use.
   ```bash
   - Finalizing and Storing Processed Data - Done -
   ```
   - To update the chat history with a new `conversations.json`, simply replace the old one and rerun the program. New messages will be processed and indexed accordingly.

## Using the Search Results

After executing a search, you'll receive a list of chat results with their titles, URLs, and creation dates. You can either:

- Use the provided URLs to view and continue the conversation directly on the ChatGPT website.
- Continue the conversation using the Chat Completions API. Enter your prompt and receive responses. You can adjust the model and temperature settings as needed in the core.py file.

Here's a sample output of the console:

```
- Search Query (0 to Exit): chromedriver update and install on ubuntu server
- Page Size (Default: 10): 3
- Search Results for 'chromedriver update and install on ubuntu server':
+---------+-----------------------------------------+---------------------+------------------------------------------------------------+
|   INDEX | TITLE                                   | CREATED AT          | URL                                                        |
+=========+=========================================+=====================+============================================================+
|       1 | Migrating Project to Ubuntu             | 2023-06-14 11:00:57 | https://chatgpt.com/c/291a78b1-7b23-474c-96c2-e123c2712263 |
+---------+-----------------------------------------+---------------------+------------------------------------------------------------+
|       2 | Using Browserless.io on Ubuntu Server   | 2023-08-02 01:02:01 | https://chatgpt.com/c/911424e0-3b94-4b89-8528-3e27c7b3fceb |
+---------+-----------------------------------------+---------------------+------------------------------------------------------------+
|       3 | ChromeDriver Error: Unexpected Argument | 2024-04-28 23:31:55 | https://chatgpt.com/c/6259c441-f5c9-493d-b49c-87d2525afeb9 |
+---------+-----------------------------------------+---------------------+------------------------------------------------------------+

- Index to Continue Chat With (0 to Return): : 1


- Title: Migrating Project to Ubuntu -
- Length: 51 Messages - Context: 13109 Tokens -
- API Input Cost: ~$0.0655+ Per Prompt Using gpt-4o Model -
- ChatGPT URL: https://chatgpt.com/c/800693d5-9715-4bf9-9d08-251126da782c -


- User: ...
-----
- Assistant: ...
.
.
.
-----
- User (0 to Return): Okay, got it fixed. One more thing, can you elaborate more on the process of updating ChromeDriver on Ubuntu?
-----
- Assistant: ...
```