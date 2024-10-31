import os
import re
import csv
from collections import defaultdict


keywords = [
    "assert", "assertEqual", "assertNotEqual", "assertTrue",
    "assertFalse", "assertIsNotNone", "assertIsNone", "assertIs",
    "assertIsNot", "assertIsInstance", "assertNotIsInstance"
]

def count_keywords_in_file(file_path, keywords):
    counts = defaultdict(int)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        for keyword in keywords:
            counts[keyword] += len(re.findall(r'\b{}\b'.format(re.escape(keyword)), content))
    return counts

def count_keywords_in_directory(directory, keywords):
    total_counts = defaultdict(int)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_counts = count_keywords_in_file(file_path, keywords)
                for keyword, count in file_counts.items():
                    total_counts[keyword] += count
    return total_counts


directory = r"./test_oracle/Shuffle"
keyword_counts = count_keywords_in_directory(directory, keywords)
total_occurrences = sum(keyword_counts.values())
output_file = r"./test_oracle\keyword_counts.csv"

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Keyword', 'Count', 'Percentage']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for keyword, count in keyword_counts.items():
        percentage = (count / total_occurrences) * 100 if total_occurrences > 0 else 0
        writer.writerow({'Keyword': keyword, 'Count': count, 'Percentage': f"{percentage:.2f}%"})

print(f"Results have been written to {output_file}")
