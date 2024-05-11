chat_models = {
    "gpt-3.5": {
        "name": "gpt-3.5-turbo",
        "version": "0125",
        "input_length": 16385,
        "output_length": 4096,
        "cost": {
            "input": 0.0005,
            "output": 0.0015,
        },
        "limits": {
            "tpm": 160000,
            "rpm": 5000,
        },
        "endpoint": "https://api.openai.com/v1/chat/completions"
    },
    "gpt-4": {
        "name": "gpt-4-turbo",
        "version": "2024-04-09",
        "input_length": 128000,
        "output_length": 8192,
        "cost": {
            "input": 0.01,
            "output": 0.03,
        },
        "limits": {
            "tpm": 600000,
            "rpm": 5000,
        },
        "endpoint": "https://api.openai.com/v1/chat/completions"
    }
}

embedding_models = {
    "large": {
        "name": "text-embedding-3-large",
        "version": "3-large",
        "input_length": 8191,
        "output_dimensions": 3072,
        "cost": {
            "input": 0.00013,
        },
        "limits": {
            "tpm": 5000000,
            "rpm": 5000,
        },
        "endpoint": "https://api.openai.com/v1/embeddings"
    },
    "small": {
        "name": "text-embedding-3-small",
        "version": "3-small",
        "input_length": 8191,
        "output_dimensions": 1536,
        "cost": {
            "input": 0.00002,
        },
        "limits": {
            "tpm": 5000000,
            "rpm": 5000,
        },
        "endpoint": "https://api.openai.com/v1/embeddings"
    }
}

models = {
    "chat": chat_models,
    "embedding": embedding_models
}
