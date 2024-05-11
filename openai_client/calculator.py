import tiktoken


class TokenCalculator:
    encoding = tiktoken.get_encoding("cl100k_base")

    def tokenize(self, context):
        return self.encoding.encode(context)

    def stringify(self, tokens):
        return self.encoding.decode(tokens)

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

    @staticmethod
    def calc_usage(usage, model):
        try:
            if usage:
                if "gpt" in model['name']:
                    tokens = {
                        f"{val}_tokens": usage[f'{key}_tokens']
                        for key, val in {
                            "prompt": "input",
                            "completion": "output",
                            "total": "total"
                        }.items()
                    }

                    costs = {
                        f"{val}_cost": usage[f'{key}_tokens'] / 1000 * model['cost'][val]
                        for key, val in {
                            "prompt": "input",
                            "completion": "output"
                        }.items()
                    }
                    costs['total_cost'] = costs['input_cost'] + costs['output_cost']

                    return {**tokens, **costs}
                elif "embedding" in model['name']:
                    return {
                        'input_tokens': usage[f'prompt_tokens'],
                        'input_cost': usage[f'prompt_tokens'] / 1000 * model['cost']['input']
                    }
            else:
                return {}
        except Exception as e:
            print(f"Error While Calculating Usage: {e}")
            return {}
