import csv

def load_examples(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        examples = [row for row in csv_reader]
    return examples

def is_consistent(hypothesis, example):
    return all(h == '?' or h == e for h, e in zip(hypothesis[:-1], example[:-1]))

def candidate_elimination(training_examples):
    G = [(['?'] * (len(training_examples[0]) - 1), 'positive')]
    S = [(['0'] * (len(training_examples[0]) - 1), 'negative')]
    
    for example in training_examples:
        if example[-1] == 'positive':
            G = [g for g in G if is_consistent(g[0], example)]
            for s in S.copy():
                if not is_consistent(s[0], example):
                    S.remove(s)
                    for i in range(len(s[0])):
                        if s[0][i] != '?' and s[0][i] != example[i]:
                            new_hypothesis = list(s[0])
                            new_hypothesis[i] = '?'
                            S.append((new_hypothesis, 'negative'))
        else:
            S = [s for s in S if not is_consistent(s[0], example)]
            for g in G.copy():
                if is_consistent(g[0], example):
                    G.remove(g)
                    for i in range(len(g[0])):
                        if g[0][i] != '?' and g[0][i] != example[i]:
                            new_hypothesis = list(g[0])
                            new_hypothesis[i] = example[i]
                            G.append((new_hypothesis, 'positive'))
    
    print("G (Most General Hypotheses):", G)
    print("S (Most Specific Hypotheses):", S)

# Load the training examples from a CSV file
training_data = load_examples("C:/Users/Acer/Desktop/ML/data.csv")  # Replace with your CSV file path
candidate_elimination(training_data)
