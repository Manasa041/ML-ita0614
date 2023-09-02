import csv
def find_s(training_data):
    positive_examples = [example for example in training_data if example[-1] == 'Yes']
    hypothesis = positive_examples[0][:-1]
    
    for instance in positive_examples:
        for i in range(len(hypothesis)):
            if hypothesis[i] != instance[i]:
                hypothesis[i] = '?' 
    return hypothesis
dataset_path = "C:/Users/Acer/Desktop/ML/enjoysport.csv"
training_data = []
with open(dataset_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        training_data.append(row)

hypothesis = find_s(training_data)
print("Final Hypothesis:", hypothesis)
