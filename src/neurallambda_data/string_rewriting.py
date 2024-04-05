import random

def generate_string(vocab, length):
    return ' '.join(random.choice(list(vocab)) for _ in range(length))

def apply_rule(tokens, rule):
    left, right = rule
    for i in range(len(tokens) - len(left) + 1):
        if tokens[i:i+len(left)] == left:
            return tokens[:i] + right + tokens[i+len(left):]
    return None

def process_string(tokens, rules):
    seen = set()
    while True:
        for rule in rules:
            new_tokens = apply_rule(tokens, rule)
            if new_tokens is not None:
                tokens_str = ' '.join(new_tokens)
                if tokens_str in seen:
                    return None  # Cycle detected
                seen.add(tokens_str)
                tokens = new_tokens
                break
        else:
            return tokens  # Fixed point reached

def generate_data(rules, min_length, max_length, num_samples):
    vocab = set()
    for left, right in rules:
        vocab.update(left)
        vocab.update(right)

    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        string = generate_string(vocab, length)
        tokens = string.split()
        result = process_string(tokens, rules)
        data.append((string, ' '.join(result) if result is not None else None))

    return data

# Example usage
rules = [
    (['A#', '#A'], []),
    (['A#', '#B'], ['#B', 'A#']),
    (['B#', '#A'], ['#A', 'B#']),
    (['B#', '#B'], []),
]

data = generate_data(rules, min_length=5, max_length=10, num_samples=10)

for string, result in data:
    if result is None:
        print(f"{string} -> Cycle detected")
    else:
        print(f"{string} -> {result}")
