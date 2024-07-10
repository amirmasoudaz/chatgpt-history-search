models = {
    "gpt-3.5-turbo": {
        "model_name": "gpt-3.5-turbo",
        "model_version": "0125",
        "context_window": 16385,
        "max_output": 4096,
        "usage_costs": {
            "input": 0.0005 / 1000,
            "output": 0.0015 / 1000
        },
        "rate_limits": {
            "tkn_per_min": 160000,
            "req_per_min": 5000
        },
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "category": "completions"
    },
    "gpt-4-turbo": {
        "model_name": "gpt-4-turbo",
        "model_version": "2024-04-09",
        "context_window": 128000,
        "max_output": 8192,
        "usage_costs": {
            "input": 0.01 / 1000,
            "output": 0.03 / 1000
        },
        "limits": {
            "tkn_per_min": 600000,
            "req_per_min": 5000
        },
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "category": "completions"
    },
    "gpt-4o": {
        "model_name": "gpt-4o",
        "model_version": "2024-05-13",
        "context_window": 128000,
        "max_output": 8192,
        "usage_costs": {
            "input": 0.005 / 1000,
            "output": 0.015 / 1000
        },
        "rate_limits": {
            "tkn_per_min": 600000,
            "req_per_min": 5000
        },
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "category": "completions"
    },
    "text-embedding-3-large": {
        "model_name": "text-embedding-3-large",
        "context_window": 8191,
        "output_dimensions": 3072,
        "usage_costs": {
            "input": 0.00013 / 1000
        },
        "rate_limits": {
            "tkn_per_min": 5000000,
            "req_per_min": 5000
        },
        "endpoint": "https://api.openai.com/v1/embeddings",
        "category": "embeddings"
    },
    "text-embedding-3-small": {
        "model_name": "text-embedding-3-small",
        "context_window": 8191,
        "output_dimensions": 1536,
        "usage_costs": {
            "input": 0.00002 / 1000
        },
        "rate_limits": {
            "tkn_per_min": 5000000,
            "req_per_min": 5000
        },
        "endpoint": "https://api.openai.com/v1/embeddings",
        "category": "embeddings"
    },
    "text_moderation_stable": {
        "model_name": "text-moderation-007",
        "context_window": 32768,
        "usage_costs": {
            "input": 0.0
        },
        "rate_limits": {
            "tkn_per_min": 150000,
            "req_per_min": 1000
        },
        "endpoint": "https://api.openai.com/v1/moderations",
        "category": "moderation"
    }
}