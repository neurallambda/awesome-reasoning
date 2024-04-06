import random

def generate_string(vocab, rules, length):
    string = []
    while len(string) < length:
        if random.random() < 0.5 and rules:
            rule = random.choice(rules)
            string.extend(rule[0])
        else:
            string.append(random.choice(list(vocab)))
    return ' '.join(string)

def apply_rule(tokens, stack, rule):
    left, right, action = rule
    for i in range(len(tokens) - len(left) + 1):
        if tokens[i:i+len(left)] == left:
            if action[0] == 'push':
                stack.append(action[1])
                return tokens[:i] + tokens[i+len(left):], stack
            elif action == 'pop' and stack:
                popped = stack.pop()
                return tokens[:i] + tokens[i+len(left):] + popped, stack
    return None, stack

def process_string(tokens, rules):
    stack = []
    seen = set()
    while True:
        for rule in rules:
            new_tokens, new_stack = apply_rule(tokens, stack, rule)
            if new_tokens is not None:
                tokens_str = ' '.join(new_tokens)
                if tokens_str in seen:
                    return None  # Cycle detected
                seen.add(tokens_str)
                tokens = new_tokens
                stack = new_stack
                break
        else:
            return tokens  # Fixed point reached

def generate_data(rules, min_length, max_length, num_samples):
    vocab = set()
    for left, _, _ in rules:
        vocab.update(left)

    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        string = generate_string(vocab, rules, length)
        tokens = string.split()
        result = process_string(tokens, rules)
        data.append((string, ' '.join(result) if result is not None else None))

    return data

# Example usage
rules = [
    (['A', 'B', 'C'], [], ('push', ['Q'])),
    # (['D', 'E', 'F'], [], ('push', ['R', 'S'])),
    (['X', 'Y', 'Z'], [], 'pop'),
    # (['M', 'N'], [], 'pop'),
]

data = generate_data(rules, min_length=5, max_length=10, num_samples=10)

for string, result in data:
    if result is None:
        print(f"{string} -> Cycle detected")
    else:
        print(f"{string} -> {result}")
