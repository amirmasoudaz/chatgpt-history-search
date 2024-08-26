import tiktoken


class Tokenizer:
    def __init__(self, model_specs: dict = None) -> None:
        """
        Initializes the OpenAI Tokenizer

        :param model_specs: dictionary of model specifications

        :return: None
        """
        if model_specs:
            self._usage_costs = model_specs["usage_costs"]
            self._model_type = model_specs["category"]
            if self._model_type == "completions":
                self.parse_usage = self._parse_completions_usage
            elif self._model_type == "embeddings":
                self.parse_usage = self._parse_embeddings_usage
            else:
                raise ValueError("Invalid model type.")
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def tokenize(self, context: str):
        return self._encoder.encode(context)

    def stringify(self, tokens: list):
        return self._encoder.decode(tokens)

    def count_tokens(self, context):
        try:
            if isinstance(context, str):
                return len(self.tokenize(context))
            elif isinstance(context, list):
                per_message = 4
                num_tokens = 0
                for message in context:
                    num_tokens += per_message
                    for key, value in message.items():
                        num_tokens += len(self.tokenize(value))
                num_tokens += 3
                return num_tokens
            else:
                return len(self.tokenize(str(context)))
        except Exception as e:
            print(f"Error While Counting Tokens: {e}")
            return 0

    def _parse_completions_usage(self, usage: dict) -> dict:
        """
        Evaluates the cost of the usage

        :param usage: Usage dictionary from the OpenAI model

        :return: dictionary of tokens and costs
        """
        if not self._usage_costs:
            raise ValueError("Usage costs not defined for the model. Pass the model_specs dictionary to the Tokenizer class")

        tokens = {
            f"{val}_tokens": usage[f'{key}_tokens']
            for key, val in {
                "prompt": "input",
                "completion": "output",
                "total": "total"
            }.items()
        }

        costs = {
            f"{val}_cost": usage[f'{key}_tokens'] / 1000 * self._usage_costs[val]
            for key, val in {
                "prompt": "input",
                "completion": "output"
            }.items()
        }
        costs['total_cost'] = costs['input_cost'] + costs['output_cost']

        return {**tokens, **costs}

    def _parse_embeddings_usage(self, usage: dict) -> dict:
        """
        Evaluates the cost of the usage

        :param usage: Usage dictionary from the OpenAI model

        :return: dictionary of tokens and costs
        """
        if not self._usage_costs:
            raise ValueError("Usage costs not defined for the model. Pass the model_specs dictionary to the Tokenizer class")

        usage = {
            'input_tokens': usage.get("prompt_tokens"),
            'input_cost': usage.get("prompt_tokens") / 1000 * self._usage_costs['input']
        }

        return usage
